import os
import json
import time
import tempfile
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import requests
import faiss
from flask import Flask, request, jsonify
from filelock import FileLock

#  CONFIG 

DATA_DIR = os.environ.get("VECTOR_DB_DIR", "./vector_db_data")
NS_DIR = os.path.join(DATA_DIR, "namespaces")

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_EMBED_URL = os.environ.get("MISTRAL_EMBED_URL", "https://api.mistral.ai/v1/embeddings")
MISTRAL_MODEL = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")  # 1024-dim text embeddings
DEFAULT_TOP_K = 5

# For mistral-embed: dim = 1024 (docs)
DEFAULT_DIM = 1024

app = Flask(__name__)


#  DISK-SAFE HELPERS 

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def atomic_write_bytes(path: str, data: bytes) -> None:
    """Write bytes atomically: write to temp, fsync, then rename."""
    ensure_dir(os.path.dirname(path))
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def atomic_write_json(path: str, obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write_bytes(path, data)

def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

#  NAMESPACE STORAGE 

def ns_paths(namespace: str) -> Dict[str, str]:
    safe_ns = namespace.strip()
    if not safe_ns:
        safe_ns = "default"
    base = os.path.join(NS_DIR, safe_ns)
    return {
        "base": base,
        "index": os.path.join(base, "index.faiss"),
        "meta": os.path.join(base, "meta.json"),
        "lock": os.path.join(base, "lock"),
    }

def now_ms() -> int:
    return int(time.time() * 1000)

def make_index(dim: int) -> faiss.Index:
    """
    Cosine similarity via Inner Product on normalized vectors.
    Use IDMap2 so we can add/remove by int64 IDs.
    """
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)

def save_index(index: faiss.Index, path: str) -> None:
    # faiss.write_index writes to a path; to be atomic, write to temp then rename.
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, path)

def load_or_create_namespace(namespace: str) -> Tuple[faiss.Index, Dict[str, Any], Dict[str, str]]:
    paths = ns_paths(namespace)
    ensure_dir(paths["base"])

    meta = read_json(paths["meta"], default={
        "dim": DEFAULT_DIM,
        "model": MISTRAL_MODEL,
        "next_int_id": 1,
        "docs": {}  # user_id -> {int_id, text, metadata, created_at, updated_at}
    })

    if os.path.exists(paths["index"]):
        index = faiss.read_index(paths["index"])
    else:
        index = make_index(int(meta["dim"]))
        save_index(index, paths["index"])
        atomic_write_json(paths["meta"], meta)

    return index, meta, paths

def persist_namespace(index: faiss.Index, meta: Dict[str, Any], paths: Dict[str, str]) -> None:
    save_index(index, paths["index"])
    atomic_write_json(paths["meta"], meta)

#  MISTRAL EMBEDDINGS 

def mistral_embed_texts(texts: List[str]) -> np.ndarray:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    payload = {
        "model": MISTRAL_MODEL,
        "input": texts
    }
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(MISTRAL_EMBED_URL, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Mistral embeddings error: {r.status_code} {r.text}")

    data = r.json()
    # Mistral returns embeddings under data[i].embedding (per docs/cookbooks)
    vectors = [item["embedding"] for item in data["data"]]
    arr = np.array(vectors, dtype="float32")

    # Normalize for cosine similarity with IP index
    faiss.normalize_L2(arr)
    return arr

def mistral_embed_one(text: str) -> np.ndarray:
    return mistral_embed_texts([text])[0:1]

#  CORE OPS 

def upsert_doc(index: faiss.Index, meta: Dict[str, Any], user_id: str, text: str, doc_meta: Dict[str, Any]) -> Dict[str, Any]:
    docs = meta["docs"]
    dim = int(meta["dim"])

    vec = mistral_embed_one(text)
    if vec.shape[1] != dim:
        raise RuntimeError(f"Embedding dim mismatch: got {vec.shape[1]} expected {dim}")

    if user_id in docs:
        # Update = remove old vector then add new
        old_int_id = int(docs[user_id]["int_id"])
        sel = faiss.IDSelectorBatch(np.array([old_int_id], dtype="int64"))
        index.remove_ids(sel)

        docs[user_id]["text"] = text
        docs[user_id]["metadata"] = doc_meta
        docs[user_id]["updated_at"] = now_ms()

        new_int_id = old_int_id
    else:
        new_int_id = int(meta["next_int_id"])
        meta["next_int_id"] = new_int_id + 1

        docs[user_id] = {
            "int_id": new_int_id,
            "text": text,
            "metadata": doc_meta,
            "created_at": now_ms(),
            "updated_at": now_ms(),
        }

    index.add_with_ids(vec, np.array([new_int_id], dtype="int64"))
    return {"id": user_id, "int_id": new_int_id}

def delete_doc(index: faiss.Index, meta: Dict[str, Any], user_id: str) -> bool:
    docs = meta["docs"]
    if user_id not in docs:
        return False
    int_id = int(docs[user_id]["int_id"])
    sel = faiss.IDSelectorBatch(np.array([int_id], dtype="int64"))
    index.remove_ids(sel)
    del docs[user_id]
    return True

def search_docs(index: faiss.Index, meta: Dict[str, Any], query: str, top_k: int) -> List[Dict[str, Any]]:
    q = mistral_embed_one(query)
    D, I = index.search(q, top_k)

    # Reverse map int_id -> user_id (we store only user_id -> int_id)
    int_to_user = {}
    for uid, rec in meta["docs"].items():
        int_to_user[int(rec["int_id"])] = uid

    results = []
    for score, int_id in zip(D[0].tolist(), I[0].tolist()):
        if int_id == -1:
            continue
        uid = int_to_user.get(int_id)
        if not uid:
            continue
        rec = meta["docs"][uid]
        results.append({
            "id": uid,
            "score": float(score),
            "text": rec["text"],
            "metadata": rec.get("metadata", {}),
            "created_at": rec.get("created_at"),
            "updated_at": rec.get("updated_at"),
        })
    return results

#  API 

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/namespaces/create")
def create_namespace():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    dim = int(body.get("dim", DEFAULT_DIM))
    model = body.get("model", MISTRAL_MODEL)

    index, meta, paths = load_or_create_namespace(namespace)
    lock = FileLock(paths["lock"])
    with lock:
        meta["dim"] = dim
        meta["model"] = model
        # If index exists with different dim, refuse (simple safety)
        if index.d != dim:
            return jsonify({"error": f"Namespace already exists with dim={index.d}, requested dim={dim}"}), 400
        persist_namespace(index, meta, paths)

    return jsonify({"namespace": namespace, "dim": dim, "model": model})

@app.get("/namespaces")
def list_namespaces():
    ensure_dir(NS_DIR)
    items = []
    for name in sorted(os.listdir(NS_DIR)):
        base = os.path.join(NS_DIR, name)
        if os.path.isdir(base):
            meta_path = os.path.join(base, "meta.json")
            meta = read_json(meta_path, default={})
            items.append({"namespace": name, "dim": meta.get("dim"), "model": meta.get("model")})
    return jsonify({"namespaces": items})

@app.post("/insert")
def insert():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    user_id = body.get("id")
    text = body.get("text", "")
    doc_meta = body.get("metadata", {}) or {}

    if not user_id or not isinstance(user_id, str):
        return jsonify({"error": "id is required (string)"}), 400
    if not text or not isinstance(text, str):
        return jsonify({"error": "text is required (string)"}), 400

    index, meta, paths = load_or_create_namespace(namespace)
    lock = FileLock(paths["lock"])

    with lock:
        try:
            out = upsert_doc(index, meta, user_id, text, doc_meta)
            persist_namespace(index, meta, paths)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"namespace": namespace, "result": out})

@app.put("/vector/<user_id>")
def update(user_id: str):
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    text = body.get("text")
    doc_meta = body.get("metadata", {}) or {}

    if text is None:
        return jsonify({"error": "text is required"}), 400

    index, meta, paths = load_or_create_namespace(namespace)
    lock = FileLock(paths["lock"])

    with lock:
        if user_id not in meta["docs"]:
            return jsonify({"error": "not found"}), 404
        try:
            out = upsert_doc(index, meta, user_id, text, doc_meta)
            persist_namespace(index, meta, paths)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"namespace": namespace, "result": out})

@app.get("/vector/<user_id>")
def get_vector(user_id: str):
    namespace = request.args.get("namespace", "default")
    _, meta, paths = load_or_create_namespace(namespace)
    lock = FileLock(paths["lock"])

    with lock:
        rec = meta["docs"].get(user_id)
        if not rec:
            return jsonify({"error": "not found"}), 404
        return jsonify({
            "namespace": namespace,
            "id": user_id,
            "int_id": rec["int_id"],
            "text": rec["text"],
            "metadata": rec.get("metadata", {}),
            "created_at": rec.get("created_at"),
            "updated_at": rec.get("updated_at"),
        })

@app.get("/vectors")
def list_vectors():
    namespace = request.args.get("namespace", "default")
    _, meta, paths = load_or_create_namespace(namespace)
    lock = FileLock(paths["lock"])

    with lock:
        items = []
        for uid, rec in meta["docs"].items():
            items.append({
                "id": uid,
                "int_id": rec["int_id"],
                "metadata": rec.get("metadata", {}),
                "created_at": rec.get("created_at"),
                "updated_at": rec.get("updated_at"),
            })
        return jsonify({"namespace": namespace, "count": len(items), "items": items})

@app.delete("/vector/<user_id>")
def delete_vector(user_id: str):
    body = request.get_json(silent=True) or {}
    namespace = body.get("namespace") or request.args.get("namespace") or "default"

    index, meta, paths = load_or_create_namespace(namespace)
    lock = FileLock(paths["lock"])

    with lock:
        ok = delete_doc(index, meta, user_id)
        if not ok:
            return jsonify({"error": "not found"}), 404
        persist_namespace(index, meta, paths)

    return jsonify({"namespace": namespace, "deleted": True, "id": user_id})

@app.post("/search")
def search():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    query = body.get("query", "")
    top_k = int(body.get("top_k", DEFAULT_TOP_K))

    if not query or not isinstance(query, str):
        return jsonify({"error": "query is required (string)"}), 400
    if top_k <= 0 or top_k > 100:
        return jsonify({"error": "top_k must be between 1 and 100"}), 400

    index, meta, paths = load_or_create_namespace(namespace)
    lock = FileLock(paths["lock"])

    with lock:
        try:
            results = search_docs(index, meta, query, top_k)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"namespace": namespace, "top_k": top_k, "results": results})

if __name__ == "__main__":
    ensure_dir(NS_DIR)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")), debug=True)
