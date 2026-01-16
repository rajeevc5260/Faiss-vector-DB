import os
import json
import time
import tempfile
from typing import Dict, Any, List, Tuple

import numpy as np
import faiss
from flask import Flask, request, jsonify
from filelock import FileLock

#  CLIP (local, free image+text embeddings)
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

#  OLLAMA (local vision-language model)
from ollama import chat

#  CONFIG 

DATA_DIR = os.environ.get("VECTOR_DB_DIR", "./vector_db_data")
NS_DIR = os.path.join(DATA_DIR, "namespaces")

# Use CLIP for BOTH text + images (one shared embedding space)
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

DEFAULT_TOP_K = 5

# CLIP ViT-L/14 embeddings dim
DEFAULT_DIM = 1024

app = Flask(__name__)

#  CLIP MODEL (LOAD ONCE) 

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()

# runs on CPU by default
# You can set device="cuda" manually if GPU is available
DEVICE = os.environ.get("CLIP_DEVICE", "cpu")
clip_model.to(DEVICE)


#  DISK-SAFE HELPERS 

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def atomic_write_bytes(path: str, data: bytes) -> None:
    ensure_dir(os.path.dirname(path))
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path),
        prefix=".tmp_",
        suffix=".bin",
    )
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
    safe_ns = namespace.strip() or "default"
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

def load_or_create_namespace(
    namespace: str,
) -> Tuple[faiss.Index, Dict[str, Any], Dict[str, str]]:
    paths = ns_paths(namespace)
    ensure_dir(paths["base"])

    meta = read_json(
        paths["meta"],
        default={
            "dim": DEFAULT_DIM,
            "model": CLIP_MODEL_NAME,
            "next_int_id": 1,
            "docs": {},
        },
    )

    if os.path.exists(paths["index"]):
        index = faiss.read_index(paths["index"])
    else:
        index = make_index(int(meta["dim"]))
        save_index(index, paths["index"])
        atomic_write_json(paths["meta"], meta)

    return index, meta, paths

def persist_namespace(
    index: faiss.Index,
    meta: Dict[str, Any],
    paths: Dict[str, str],
) -> None:
    save_index(index, paths["index"])
    atomic_write_json(paths["meta"], meta)

#  CLIP EMBEDDINGS 

def clip_embed_text(text: str) -> np.ndarray:
    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        vec = clip_model.get_text_features(**inputs)

    vec = vec / vec.norm(p=2, dim=-1, keepdim=True)
    return vec.cpu().numpy().astype("float32")

def clip_embed_image(file_storage) -> np.ndarray:
    img = Image.open(file_storage.stream).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        vec = clip_model.get_image_features(**inputs)

    vec = vec / vec.norm(p=2, dim=-1, keepdim=True)
    return vec.cpu().numpy().astype("float32")

#  IMAGE → TEXT (OLLAMA)

def extract_image_keywords(file_storage) -> str:
    """
    Uses local Ollama Qwen3-VL to extract descriptive keywords for an image.
    Returns a short comma-separated text.
    """
    img = Image.open(file_storage.stream).convert("RGB")

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as tmp:
        img.save(tmp.name)
        image_path = tmp.name

    try:
        res = chat(
            model="qwen3-vl:2b",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Describe this image using clear short phrases only. "
                        "Focus on objects, actions, colors, environment and should be meaningful. "
                        "Return comma separated only."
                    ),
                    "images": [image_path],
                }
            ],
        )

        return res["message"]["content"].strip()

    finally:
        try:
            os.remove(image_path)
        except Exception:
            pass

#  CORE OPS 

def upsert_vector(
    index: faiss.Index,
    meta: Dict[str, Any],
    user_id: str,
    vec: np.ndarray,
    doc_type: str,
    doc_meta: Dict[str, Any],
    text_optional: str = "",
) -> Dict[str, Any]:
    docs = meta["docs"]
    dim = int(meta["dim"])

    if vec.shape[1] != dim:
        raise RuntimeError(
            f"Embedding dim mismatch: got {vec.shape[1]} expected {dim}"
        )

    if user_id in docs:
        old_int_id = int(docs[user_id]["int_id"])
        sel = faiss.IDSelectorBatch(
            np.array([old_int_id], dtype="int64")
        )
        index.remove_ids(sel)

        docs[user_id]["type"] = doc_type
        docs[user_id]["metadata"] = doc_meta
        if text_optional:
            docs[user_id]["text"] = text_optional
        docs[user_id]["updated_at"] = now_ms()

        new_int_id = old_int_id
    else:
        new_int_id = int(meta["next_int_id"])
        meta["next_int_id"] = new_int_id + 1

        docs[user_id] = {
            "int_id": new_int_id,
            "type": doc_type,
            "text": text_optional or "",
            "metadata": doc_meta,
            "created_at": now_ms(),
            "updated_at": now_ms(),
        }

    index.add_with_ids(
        vec,
        np.array([new_int_id], dtype="int64"),
    )

    return {"id": user_id, "int_id": new_int_id}

def delete_doc(
    index: faiss.Index,
    meta: Dict[str, Any],
    user_id: str,
) -> bool:
    docs = meta["docs"]
    if user_id not in docs:
        return False

    int_id = int(docs[user_id]["int_id"])
    sel = faiss.IDSelectorBatch(
        np.array([int_id], dtype="int64")
    )
    index.remove_ids(sel)
    del docs[user_id]
    return True

def search_docs(
    index: faiss.Index,
    meta: Dict[str, Any],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    q = clip_embed_text(query)
    D, I = index.search(q, top_k * 2)

    int_to_user = {
        int(rec["int_id"]): uid
        for uid, rec in meta["docs"].items()
    }

    results = []
    for score, int_id in zip(D[0].tolist(), I[0].tolist()):
        if int_id == -1:
            continue

        uid = int_to_user.get(int_id)
        if not uid:
            continue

        rec = meta["docs"][uid]

        boost = 0.05 if rec.get("type") == "image" else 0.0

        results.append({
            "id": uid,
            "score": float(score + boost),
            "type": rec.get("type"),
            "text": rec.get("text", ""),
            "metadata": rec.get("metadata", {}),
            "created_at": rec.get("created_at"),
            "updated_at": rec.get("updated_at"),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

#  API 

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/insert")
def insert():
    ct = (request.content_type or "").lower()

    #  multipart (image OR text)
    if "multipart/form-data" in ct:
        namespace = request.form.get("namespace", "default")
        user_id = request.form.get("id")
        image = request.files.get("image")
        text = request.form.get("text")
        doc_meta_raw = request.form.get("metadata")
        doc_meta = json.loads(doc_meta_raw) if doc_meta_raw else {}

        if not user_id or not isinstance(user_id, str):
            return jsonify({"error": "id is required (string)"}), 400
        if image is None and not text:
            return jsonify({"error": "provide image or text"}), 400

        index, meta, paths = load_or_create_namespace(namespace)
        lock = FileLock(paths["lock"])

        with lock:
            try:
                if image is not None:
                    keywords = extract_image_keywords(image)
                    vec = clip_embed_image(image)

                    doc_meta["filename"] = image.filename
                    doc_meta["image_keywords"] = keywords

                    out = upsert_vector(
                        index,
                        meta,
                        user_id,
                        vec,
                        "image",
                        doc_meta,
                        text_optional=keywords,
                    )
                else:
                    vec = clip_embed_text(text)
                    out = upsert_vector(
                        index,
                        meta,
                        user_id,
                        vec,
                        "text",
                        doc_meta,
                        text_optional=text,
                    )

                persist_namespace(index, meta, paths)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return jsonify({"namespace": namespace, "result": out})

    #  JSON (text-only)
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
            vec = clip_embed_text(text)
            out = upsert_vector(
                index,
                meta,
                user_id,
                vec,
                "text",
                doc_meta,
                text_optional=text,
            )
            persist_namespace(index, meta, paths)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"namespace": namespace, "result": out})


@app.post("/search")
def search():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    query = body.get("query", "")
    top_k = int(body.get("top_k", DEFAULT_TOP_K))

    if not query:
        return jsonify({"error": "query required"}), 400

    index, meta, paths = load_or_create_namespace(namespace)
    lock = FileLock(paths["lock"])

    with lock:
        results = search_docs(index, meta, query, top_k)

    return jsonify({"namespace": namespace, "results": results})

@app.post("/search/image")
def search_by_image():
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    namespace = request.form.get("namespace", "default")
    top_k = int(request.form.get("top_k", DEFAULT_TOP_K))
    image = request.files["image"]

    index, meta, paths = load_or_create_namespace(namespace)
    lock = FileLock(paths["lock"])

    with lock:
        vec = clip_embed_image(image)
        D, I = index.search(vec, top_k)

        int_to_user = {
            int(rec["int_id"]): uid
            for uid, rec in meta["docs"].items()
        }

        results = []
        for score, int_id in zip(D[0].tolist(), I[0].tolist()):
            uid = int_to_user.get(int_id)
            if not uid:
                continue

            rec = meta["docs"][uid]
            results.append({
                "id": uid,
                "score": float(score),
                "type": rec.get("type"),
                "metadata": rec.get("metadata", {}),
            })

    return jsonify({"namespace": namespace, "results": results})

if __name__ == "__main__":
    ensure_dir(NS_DIR)
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)

