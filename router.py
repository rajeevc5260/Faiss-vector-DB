import os
import json
import hashlib
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Example: "http://localhost:6001,http://localhost:6002"
SHARD_URLS = os.environ.get("SHARD_URLS", "http://localhost:6001,http://localhost:6002").split(",")
ROUTER_PORT = int(os.environ.get("PORT", "5001"))

DEFAULT_TOP_K = int(os.environ.get("TOP_K", "5"))

def stable_shard_for_id(doc_id: str) -> int:
    h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    return int(h, 16) % len(SHARD_URLS)

def merge_results(all_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    best = {}
    for r in all_results:
        rid = r["id"]
        if rid not in best or r["score"] > best[rid]["score"]:
            best[rid] = r
    out = list(best.values())
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]

@app.get("/health")
def health():
    # basic shard health check
    statuses = []
    for u in SHARD_URLS:
        try:
            r = requests.get(f"{u}/health", timeout=2)
            statuses.append({"url": u, "ok": r.ok, "json": r.json() if r.ok else None})
        except Exception as e:
            statuses.append({"url": u, "ok": False, "error": str(e)})
    return jsonify({"ok": True, "router": True, "shards": statuses})

@app.post("/insert")
def insert():
    ct = (request.content_type or "").lower()

    # multipart (image or text)
    if "multipart/form-data" in ct:
        namespace = request.form.get("namespace", "default")
        doc_id = request.form.get("id", "")
        if not doc_id:
            return jsonify({"error": "id is required"}), 400

        shard_idx = stable_shard_for_id(doc_id)
        url = f"{SHARD_URLS[shard_idx]}/insert"

        data = dict(request.form)
        files = {}
        if "image" in request.files:
            f = request.files["image"]
            files["image"] = (f.filename, f.stream, f.mimetype)

        try:
            resp = requests.post(url, data=data, files=files, timeout=120)
            return jsonify({"routed_shard": shard_idx, "shard_url": SHARD_URLS[shard_idx], "response": resp.json()}), resp.status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # JSON (text)
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    doc_id = body.get("id", "")
    if not doc_id:
        return jsonify({"error": "id is required"}), 400

    shard_idx = stable_shard_for_id(doc_id)
    url = f"{SHARD_URLS[shard_idx]}/insert"

    try:
        resp = requests.post(url, json=body, timeout=120)
        return jsonify({"routed_shard": shard_idx, "shard_url": SHARD_URLS[shard_idx], "response": resp.json()}), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/search")
def search():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    query = body.get("query", "")
    top_k = int(body.get("top_k", DEFAULT_TOP_K))
    nprobe = body.get("nprobe")

    if not query:
        return jsonify({"error": "query required"}), 400

    # fanout to all shards in parallel
    def call_shard(url: str):
        payload = {"namespace": namespace, "query": query, "top_k": top_k}
        if nprobe is not None:
            payload["nprobe"] = nprobe
        r = requests.post(f"{url}/search", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    results = []
    shard_infos = []
    with ThreadPoolExecutor(max_workers=len(SHARD_URLS)) as ex:
        futs = {ex.submit(call_shard, u): u for u in SHARD_URLS}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                js = fut.result()
                shard_infos.append({"url": u, "trained": js.get("trained"), "shard_id": js.get("shard_id")})
                results.extend(js.get("results", []))
            except Exception as e:
                shard_infos.append({"url": u, "error": str(e)})

    merged = merge_results(results, top_k)
    return jsonify({"namespace": namespace, "shards": shard_infos, "results": merged})

@app.post("/search/image")
def search_image():
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    namespace = request.form.get("namespace", "default")
    top_k = int(request.form.get("top_k", DEFAULT_TOP_K))
    nprobe = request.form.get("nprobe")

    # read the uploaded file once into memory for re-sending to shards
    f = request.files["image"]
    img_bytes = f.read()
    filename = f.filename or "query.png"
    mimetype = f.mimetype or "image/png"

    def call_shard(url: str):
        data = {"namespace": namespace, "top_k": str(top_k)}
        if nprobe is not None:
            data["nprobe"] = str(nprobe)
        files = {"image": (filename, img_bytes, mimetype)}
        r = requests.post(f"{url}/search/image", data=data, files=files, timeout=120)
        r.raise_for_status()
        return r.json()

    results = []
    shard_infos = []
    with ThreadPoolExecutor(max_workers=len(SHARD_URLS)) as ex:
        futs = {ex.submit(call_shard, u): u for u in SHARD_URLS}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                js = fut.result()
                shard_infos.append({"url": u, "trained": js.get("trained"), "shard_id": js.get("shard_id")})
                results.extend(js.get("results", []))
            except Exception as e:
                shard_infos.append({"url": u, "error": str(e)})

    merged = merge_results(results, top_k)
    return jsonify({"namespace": namespace, "shards": shard_infos, "results": merged})

@app.post("/admin/retrain")
def retrain():
    """
    Fan-out retrain request to all shards.
    Actual training happens once due to shared lock.
    """
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    last_n = body.get("last_n")

    def call_shard(url: str):
        payload = {"namespace": namespace}
        if last_n is not None:
            payload["last_n"] = last_n

        r = requests.post(f"{url}/admin/retrain", json=payload, timeout=600)
        return {
            "url": url,
            "status_code": r.status_code,
            "response": r.json() if r.content else None
        }

    results = []
    with ThreadPoolExecutor(max_workers=len(SHARD_URLS)) as ex:
        futs = {ex.submit(call_shard, u): u for u in SHARD_URLS}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"error": str(e)})

    return jsonify({
        "namespace": namespace,
        "fanout": True,
        "shards": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=ROUTER_PORT, debug=False, use_reloader=False)
