import os
import json
import hashlib
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    resp.headers["Access-Control-Allow-Origin"] = origin or "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Max-Age"] = "86400"
    resp.headers["Vary"] = "Origin"
    return resp

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return add_cors_headers(app.make_default_options_response())
    return None

@app.after_request
def attach_cors(resp):
    return add_cors_headers(resp)

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

@app.delete("/delete")
def delete_doc():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    doc_id = body.get("id", "")
    if not doc_id:
        return jsonify({"error": "id is required"}), 400

    shard_idx = stable_shard_for_id(doc_id)
    url = f"{SHARD_URLS[shard_idx]}/delete"

    try:
        resp = requests.delete(url, json=body, timeout=60)
        return jsonify({"routed_shard": shard_idx, "shard_url": SHARD_URLS[shard_idx], "response": resp.json()}), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@app.post("/admin/namespace/create")
def create_namespace():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    if not namespace or not isinstance(namespace, str):
        return jsonify({"error": "namespace is required (string)"}), 400

    def call_shard(url: str):
        payload = {"namespace": namespace}
        r = requests.post(f"{url}/admin/namespace/create", json=payload, timeout=30)
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

@app.get("/admin/namespaces")
def get_namespaces():
    def call_shard(url: str):
        r = requests.get(f"{url}/admin/namespaces", timeout=30)
        return {
            "url": url,
            "status_code": r.status_code,
            "response": r.json() if r.content else None
        }

    results = []
    names = set()
    with ThreadPoolExecutor(max_workers=len(SHARD_URLS)) as ex:
        futs = {ex.submit(call_shard, u): u for u in SHARD_URLS}
        for fut in as_completed(futs):
            try:
                res = fut.result()
                results.append(res)
                resp = res.get("response") or {}
                for n in resp.get("namespaces", []) or []:
                    names.add(n)
            except Exception as e:
                results.append({"error": str(e)})

    return jsonify({
        "fanout": True,
        "namespaces": sorted(names),
        "shards": results
    })

@app.get("/admin/namespace")
def get_namespace():
    namespace = request.args.get("namespace", "default")
    include_training_stats = request.args.get("include_training_stats")

    def call_shard(url: str):
        params = {"namespace": namespace}
        if include_training_stats is not None:
            params["include_training_stats"] = include_training_stats
        r = requests.get(f"{url}/admin/namespace", params=params, timeout=30)
        return {
            "url": url,
            "status_code": r.status_code,
            "response": r.json() if r.content else None
        }

    results = []
    totals = {"docs_total": 0, "docs_in_index": 0, "staging_vectors": 0, "training_pool": 0}
    with ThreadPoolExecutor(max_workers=len(SHARD_URLS)) as ex:
        futs = {ex.submit(call_shard, u): u for u in SHARD_URLS}
        for fut in as_completed(futs):
            try:
                res = fut.result()
                results.append(res)
                resp = res.get("response") or {}
                counts = resp.get("counts") or {}
                for k in totals:
                    if k == "training_pool":
                        totals[k] = max(totals[k], int(counts.get(k, 0)))
                    else:
                        totals[k] += int(counts.get(k, 0))
            except Exception as e:
                results.append({"error": str(e)})

    return jsonify({
        "namespace": namespace,
        "fanout": True,
        "totals": totals,
        "shards": results
    })

@app.get("/admin/namespace/docs")
def list_namespace_docs():
    namespace = request.args.get("namespace", "default")
    shard = request.args.get("shard", "all")
    limit = request.args.get("limit", "50")
    after_id = request.args.get("after_id")
    include_text = request.args.get("include_text")
    include_metadata = request.args.get("include_metadata")
    include_total = request.args.get("include_total")

    def call_shard(url: str):
        params = {
            "namespace": namespace,
            "limit": limit,
        }
        if after_id is not None:
            params["after_id"] = after_id
        if include_text is not None:
            params["include_text"] = include_text
        if include_metadata is not None:
            params["include_metadata"] = include_metadata
        if include_total is not None:
            params["include_total"] = include_total
        r = requests.get(f"{url}/admin/namespace/docs", params=params, timeout=30)
        return {
            "url": url,
            "status_code": r.status_code,
            "response": r.json() if r.content else None
        }

    if shard != "all":
        try:
            shard_idx = int(shard)
        except ValueError:
            return jsonify({"error": "shard must be an integer or 'all'"}), 400
        if shard_idx < 0 or shard_idx >= len(SHARD_URLS):
            return jsonify({"error": "shard out of range"}), 400
        res = call_shard(SHARD_URLS[shard_idx])
        return jsonify({
            "namespace": namespace,
            "shard": shard_idx,
            "response": res
        }), res.get("status_code", 200)

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
        "mode": "per_shard",
        "shards": results
    })

@app.delete("/admin/namespace")
def delete_namespace():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    if not namespace or not isinstance(namespace, str):
        return jsonify({"error": "namespace is required (string)"}), 400

    def call_shard(url: str):
        payload = {"namespace": namespace}
        r = requests.delete(f"{url}/admin/namespace", json=payload, timeout=30)
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
