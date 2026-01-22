import os
import json
import time
import base64
import sqlite3
import tempfile
import hashlib
import shutil
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import faiss
from flask import Flask, request, jsonify
from filelock import FileLock

# CLIP (local embeddings)
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# OLLAMA (local vision-language model)
from ollama import chat



# CONFIG


DATA_DIR = os.environ.get("VECTOR_DB_DIR", "./vector_db_data")
NS_DIR = os.path.join(DATA_DIR, "namespaces")

CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
DEFAULT_DIM = int(os.environ.get("VECTOR_DIM", "1024"))
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "5"))

# Shard identity
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
PORT = int(os.environ.get("PORT", "6001"))

# Training / IVF+PQ params (prototype defaults)
TRAIN_MIN = int(os.environ.get("TRAIN_MIN", "256"))          # small for prototype; production 50k-200k+
TRAIN_SAMPLE_MAX = int(os.environ.get("TRAIN_SAMPLE_MAX", "5000"))  # cap for training speed on laptop

# IVF + PQ params
# For real scale: NLIST 4096+, M=64, NBITS=8 (your earlier targets)
NLIST = int(os.environ.get("IVF_NLIST", "256"))   # small for prototype; use 4096 for 1M scale
PQ_M = int(os.environ.get("PQ_M", "32"))          # prototype; use 64 for 1024-dim
PQ_NBITS = int(os.environ.get("PQ_NBITS", "8"))

# Search tuning
DEFAULT_NPROBE = int(os.environ.get("NPROBE", "16"))
OVERSAMPLE = int(os.environ.get("OVERSAMPLE", "10"))  # shard returns top_k*OVERSAMPLE, router merges

# Device
DEVICE = os.environ.get("CLIP_DEVICE", "cpu")



# APP


app = Flask(__name__)

# CORS headers for browser-based admin UI calls.
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

# In-memory FAISS FlatIP index for staging vectors (per namespace).
STAGING_FLAT_INDEXES: Dict[str, faiss.Index] = {}

# Load CLIP once per shard process
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()
clip_model.to(DEVICE)



# DISK HELPERS


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_ms() -> int:
    return int(time.time() * 1000)

def atomic_write_bytes(path: str, data: bytes) -> None:
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
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}



# PATHS (namespace + shard)


def ns_base(namespace: str) -> str:
    safe_ns = namespace.strip() or "default"
    return os.path.join(NS_DIR, safe_ns)

def ns_training_dir(namespace: str) -> str:
    return os.path.join(ns_base(namespace), "training")

def ns_training_pool_db(namespace: str) -> str:
    return os.path.join(ns_training_dir(namespace), "pool.db")

def ns_training_lock(namespace: str) -> str:
    return os.path.join(ns_training_dir(namespace), "train.lock")

def ns_training_template(namespace: str) -> str:
    return os.path.join(ns_training_dir(namespace), "ivfpq_template.faiss")

def ns_training_stats(namespace: str) -> str:
    return os.path.join(ns_training_dir(namespace), "train_stats.json")

def shard_dir(namespace: str) -> str:
    return os.path.join(ns_base(namespace), "shards", f"shard_{SHARD_ID}")

def shard_index_path(namespace: str) -> str:
    return os.path.join(shard_dir(namespace), "index.faiss")

def shard_state_path(namespace: str) -> str:
    return os.path.join(shard_dir(namespace), "state.json")

def shard_lock_path(namespace: str) -> str:
    return os.path.join(shard_dir(namespace), "lock")

def shard_meta_db(namespace: str) -> str:
    return os.path.join(shard_dir(namespace), "meta.db")

def shard_staging_db(namespace: str) -> str:
    return os.path.join(shard_dir(namespace), "staging.db")

def list_namespaces() -> List[str]:
    ensure_dir(NS_DIR)
    out = []
    for name in os.listdir(NS_DIR):
        path = os.path.join(NS_DIR, name)
        if os.path.isdir(path):
            out.append(name)
    out.sort()
    return out



# SQLITE (meta + staging + training pool)


def sqlite_connect(path: str) -> sqlite3.Connection:
    ensure_dir(os.path.dirname(path))
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_shard_dbs(namespace: str) -> None:
    # meta.db
    con = sqlite_connect(shard_meta_db(namespace))
    con.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id TEXT PRIMARY KEY,
            faiss_id INTEGER,
            in_index INTEGER NOT NULL,   -- 1 => in FAISS IVF index, 0 => in staging only
            type TEXT NOT NULL,
            text TEXT,
            metadata_json TEXT,
            created_at INTEGER,
            updated_at INTEGER
        );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_docs_faiss_id ON docs(faiss_id);")
    con.commit()
    con.close()

    # staging.db
    con = sqlite_connect(shard_staging_db(namespace))
    ensure_staging_schema(con)
    con.commit()
    con.close()

    # training pool db (shared per namespace)
    ensure_dir(ns_training_dir(namespace))
    con = sqlite_connect(ns_training_pool_db(namespace))
    ensure_training_pool_schema(con)
    con.commit()
    con.close()

def count_meta_docs(namespace: str) -> Tuple[int, int]:
    con = sqlite_connect(shard_meta_db(namespace))
    total = int(con.execute("SELECT COUNT(*) FROM docs;").fetchone()[0])
    in_index = int(con.execute("SELECT COUNT(*) FROM docs WHERE in_index=1;").fetchone()[0])
    con.close()
    return total, in_index

def count_staging_vectors(namespace: str) -> int:
    con = sqlite_connect(shard_staging_db(namespace))
    ensure_staging_schema(con)
    n = int(con.execute("SELECT COUNT(*) FROM staging_vectors;").fetchone()[0])
    con.close()
    return n

def ensure_staging_schema(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS staging_vectors (
            id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            created_at INTEGER
        );
    """)

def ensure_training_pool_schema(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS training_pool (
            uid TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            created_at INTEGER
        );
    """)



# CLIP EMBEDDINGS


def clip_embed_text(text: str) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        vec = clip_model.get_text_features(**inputs)
    vec = vec / vec.norm(p=2, dim=-1, keepdim=True)
    return vec.cpu().numpy().astype("float32")  # shape (1, dim)

def clip_embed_image(file_storage) -> np.ndarray:
    img = Image.open(file_storage.stream).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vec = clip_model.get_image_features(**inputs)
    vec = vec / vec.norm(p=2, dim=-1, keepdim=True)
    return vec.cpu().numpy().astype("float32")  # shape (1, dim)

def extract_image_keywords(file_storage) -> str:
    img = Image.open(file_storage.stream).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        image_path = tmp.name
    try:
        res = chat(
            model="qwen3-vl:2b",
            messages=[{
                "role": "user",
                "content": (
                    "Describe this image using clear short phrases only. "
                    "Focus on objects, actions, colors, environment and should be meaningful. "
                    "Return comma separated only."
                ),
                "images": [image_path],
            }],
        )
        return (res.get("message", {}) or {}).get("content", "").strip()
    finally:
        try:
            os.remove(image_path)
        except Exception:
            pass



# FAISS IVF+PQ (trained template + per-shard index)


def build_ivfpq_template(dim: int) -> faiss.Index:
    # Quantizer defines coarse centroids. Use IP since vectors are normalized => cosine-like.
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFPQ(quantizer, dim, NLIST, PQ_M, PQ_NBITS, faiss.METRIC_INNER_PRODUCT)
    # Wrap with IDMap2 so we can remove by int ids
    return faiss.IndexIDMap2(ivf)

def is_trained_template_exists(namespace: str) -> bool:
    return os.path.exists(ns_training_template(namespace))

def load_or_create_shard_state(namespace: str) -> Dict[str, Any]:
    path = shard_state_path(namespace)
    state = read_json(path, default={
        "dim": DEFAULT_DIM,
        "next_faiss_id": 1,
        "trained_ready": False,  # shard has bootstrapped its index from template
        "created_at": now_ms(),
        "updated_at": now_ms(),
    })
    return state

def save_shard_state(namespace: str, state: Dict[str, Any]) -> None:
    state["updated_at"] = now_ms()
    atomic_write_json(shard_state_path(namespace), state)

def set_nprobe(index: faiss.Index, nprobe: int) -> None:
    try:
        # IndexIDMap2 wraps underlying index in .index
        if hasattr(index, "index") and hasattr(index.index, "nprobe"):
            index.index.nprobe = int(nprobe)
    except Exception:
        pass

def save_faiss_index(index: faiss.Index, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, path)

def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

def ensure_shard_index_ready(namespace: str) -> Tuple[Optional[faiss.Index], Dict[str, Any]]:
    """
    Returns (index_or_none, shard_state).
    If template exists and shard has not bootstrapped, it will bootstrap and ingest staging.
    """
    init_shard_dbs(namespace)
    state = load_or_create_shard_state(namespace)

    template_exists = is_trained_template_exists(namespace)
    if not template_exists:
        # No training done yet, no IVF index
        return None, state

    # Template exists: ensure shard has a FAISS index file
    idx_path = shard_index_path(namespace)
    if os.path.exists(idx_path):
        index = load_faiss_index(idx_path)
    else:
        # Start from trained template (empty index with trained params)
        index = load_faiss_index(ns_training_template(namespace))
        save_faiss_index(index, idx_path)

    # Mark shard trained_ready (means index exists and trained)
    state["trained_ready"] = True
    save_shard_state(namespace, state)

    return index, state



# TRAINING POOL (shared per namespace)


def pool_count(namespace: str) -> int:
    con = sqlite_connect(ns_training_pool_db(namespace))
    ensure_training_pool_schema(con)
    cur = con.execute("SELECT COUNT(*) FROM training_pool;")
    n = int(cur.fetchone()[0])
    con.close()
    return n

def pool_add_vector(namespace: str, uid: str, vec_1xd: np.ndarray) -> None:
    # store raw float32 bytes (dim*4)
    vec = vec_1xd.reshape(-1).astype("float32")
    con = sqlite_connect(ns_training_pool_db(namespace))
    ensure_training_pool_schema(con)
    con.execute(
        "INSERT OR REPLACE INTO training_pool(uid, vector, created_at) VALUES(?,?,?)",
        (uid, vec.tobytes(), now_ms())
    )
    con.commit()
    con.close()

def pool_delete_vector(namespace: str, uid: str) -> None:
    con = sqlite_connect(ns_training_pool_db(namespace))
    ensure_training_pool_schema(con)
    con.execute("DELETE FROM training_pool WHERE uid=?", (uid,))
    con.commit()
    con.close()

def pool_sample_vectors(namespace: str, max_n: int) -> np.ndarray:
    con = sqlite_connect(ns_training_pool_db(namespace))
    ensure_training_pool_schema(con)
    cur = con.execute("SELECT vector FROM training_pool ORDER BY created_at DESC LIMIT ?", (max_n,))
    rows = cur.fetchall()
    con.close()
    if not rows:
        return np.zeros((0, DEFAULT_DIM), dtype="float32")
    X = np.stack([np.frombuffer(r[0], dtype="float32") for r in rows], axis=0)
    return X.astype("float32")

def run_namespace_training_if_needed(
    namespace: str,
    dim: int,
    force: bool = False,        # <-- manual retrain flag
    sample_limit: Optional[int] = None  # <-- last N vectors
) -> bool:
    """
    Train IVF+PQ model.
    - Runs automatically only if not trained yet
    - Can be forced manually later
    """

    ensure_dir(ns_training_dir(namespace))
    train_lock = FileLock(ns_training_lock(namespace))

    with train_lock:
        already_trained = is_trained_template_exists(namespace)

        if already_trained and not force:
            return True

        n = pool_count(namespace)
        if n < TRAIN_MIN:
            return False

        # Decide how many vectors to train on
        take_n = sample_limit or min(n, TRAIN_SAMPLE_MAX)

        X = pool_sample_vectors(namespace, take_n)
        if X.shape[0] < TRAIN_MIN:
            return False

        print(
            f"[TRAINING START] namespace={namespace} | "
            f"vectors_used={X.shape[0]} | "
            f"total_pool={n} | "
            f"nlist={NLIST} | pq_m={PQ_M} | pq_nbits={PQ_NBITS}"
        )

        index = build_ivfpq_template(dim)
        index.train(X)

        save_faiss_index(index, ns_training_template(namespace))

        atomic_write_json(ns_training_stats(namespace), {
            "trained_at": now_ms(),
            "dim": dim,
            "train_vectors_used": int(X.shape[0]),
            "total_pool_vectors": n,
            "nlist": NLIST,
            "pq_m": PQ_M,
            "pq_nbits": PQ_NBITS,
            "forced": force,
        })

        print(f"[TRAINING DONE] namespace={namespace}")

        return True




# SHARD STORAGE OPS (staging + meta + IVF index)


def meta_upsert_doc(namespace: str, doc_id: str, faiss_id: int, in_index: int,
                    doc_type: str, text: str, metadata: Dict[str, Any]) -> None:
    con = sqlite_connect(shard_meta_db(namespace))
    existing = con.execute("SELECT id, created_at FROM docs WHERE id=?", (doc_id,)).fetchone()
    created_at = existing[1] if existing else now_ms()
    con.execute("""
        INSERT OR REPLACE INTO docs(id, faiss_id, in_index, type, text, metadata_json, created_at, updated_at)
        VALUES(?,?,?,?,?,?,?,?)
    """, (
        doc_id,
        int(faiss_id),
        int(in_index),
        doc_type,
        text or "",
        json.dumps(metadata or {}, ensure_ascii=False),
        int(created_at),
        now_ms(),
    ))
    con.commit()
    con.close()

def meta_get_doc(namespace: str, doc_id: str) -> Optional[Dict[str, Any]]:
    con = sqlite_connect(shard_meta_db(namespace))
    row = con.execute("""
        SELECT id, faiss_id, in_index, type, text, metadata_json, created_at, updated_at
        FROM docs WHERE id=?
    """, (doc_id,)).fetchone()
    con.close()
    if not row:
        return None
    return {
        "id": row[0],
        "faiss_id": int(row[1]) if row[1] is not None else None,
        "in_index": int(row[2]),
        "type": row[3],
        "text": row[4] or "",
        "metadata": json.loads(row[5] or "{}"),
        "created_at": row[6],
        "updated_at": row[7],
    }

def meta_get_faiss_ids(namespace: str, doc_ids: List[str]) -> Dict[str, int]:
    if not doc_ids:
        return {}
    out: Dict[str, int] = {}
    con = sqlite_connect(shard_meta_db(namespace))
    batch = 900  # keep under sqlite var limit
    for i in range(0, len(doc_ids), batch):
        chunk = doc_ids[i:i+batch]
        placeholders = ",".join(["?"] * len(chunk))
        rows = con.execute(
            f"SELECT id, faiss_id FROM docs WHERE id IN ({placeholders})",
            chunk
        ).fetchall()
        for doc_id, faiss_id in rows:
            if faiss_id is not None:
                out[doc_id] = int(faiss_id)
    con.close()
    return out

def meta_delete_doc(namespace: str, doc_id: str) -> None:
    con = sqlite_connect(shard_meta_db(namespace))
    con.execute("DELETE FROM docs WHERE id=?", (doc_id,))
    con.commit()
    con.close()

def build_staging_flat_index(namespace: str, dim: int) -> faiss.Index:
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    staged = staging_all_vectors(namespace)
    if not staged:
        return index

    doc_ids = [doc_id for doc_id, _ in staged]
    faiss_ids = meta_get_faiss_ids(namespace, doc_ids)

    vecs = []
    ids = []
    for doc_id, vec in staged:
        fid = faiss_ids.get(doc_id)
        if fid is None:
            continue
        vecs.append(vec.reshape(1, -1))
        ids.append(fid)

    if ids:
        X = np.vstack(vecs).astype("float32")
        I = np.array(ids, dtype="int64")
        index.add_with_ids(X, I)
    return index

def get_or_build_staging_index(namespace: str, dim: int) -> faiss.Index:
    index = STAGING_FLAT_INDEXES.get(namespace)
    if index is None:
        index = build_staging_flat_index(namespace, dim)
        STAGING_FLAT_INDEXES[namespace] = index
    return index

def staging_index_add(namespace: str, faiss_id: int, vec_1xd: np.ndarray) -> None:
    index = STAGING_FLAT_INDEXES.get(namespace)
    if index is None:
        return
    index.add_with_ids(vec_1xd.reshape(1, -1).astype("float32"), np.array([int(faiss_id)], dtype="int64"))

def staging_index_remove(namespace: str, faiss_id: int) -> None:
    index = STAGING_FLAT_INDEXES.get(namespace)
    if index is None:
        return
    sel = faiss.IDSelectorBatch(np.array([int(faiss_id)], dtype="int64"))
    index.remove_ids(sel)

def reset_staging_index(namespace: str, dim: int) -> None:
    if namespace not in STAGING_FLAT_INDEXES:
        return
    STAGING_FLAT_INDEXES[namespace] = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

def staging_put(namespace: str, doc_id: str, vec_1xd: np.ndarray, faiss_id: Optional[int] = None) -> None:
    vec = vec_1xd.reshape(-1).astype("float32")
    con = sqlite_connect(shard_staging_db(namespace))
    ensure_staging_schema(con)
    con.execute(
        "INSERT OR REPLACE INTO staging_vectors(id, vector, created_at) VALUES(?,?,?)",
        (doc_id, vec.tobytes(), now_ms())
    )
    con.commit()
    con.close()

    if faiss_id is None:
        rec = meta_get_doc(namespace, doc_id)
        faiss_id = rec["faiss_id"] if rec else None
    if faiss_id is not None:
        staging_index_add(namespace, faiss_id, vec_1xd)

def staging_delete(namespace: str, doc_id: str, faiss_id: Optional[int] = None, remove_from_mem: bool = True) -> None:
    con = sqlite_connect(shard_staging_db(namespace))
    ensure_staging_schema(con)
    con.execute("DELETE FROM staging_vectors WHERE id=?", (doc_id,))
    con.commit()
    con.close()

    if remove_from_mem:
        if faiss_id is None:
            rec = meta_get_doc(namespace, doc_id)
            faiss_id = rec["faiss_id"] if rec else None
        if faiss_id is not None:
            staging_index_remove(namespace, faiss_id)

def staging_all_vectors(namespace: str) -> List[Tuple[str, np.ndarray]]:
    con = sqlite_connect(shard_staging_db(namespace))
    ensure_staging_schema(con)
    rows = con.execute("SELECT id, vector FROM staging_vectors").fetchall()
    con.close()
    out = []
    for doc_id, blob in rows:
        out.append((doc_id, np.frombuffer(blob, dtype="float32")))
    return out

def allocate_faiss_id(state: Dict[str, Any]) -> int:
    fid = int(state["next_faiss_id"])
    state["next_faiss_id"] = fid + 1
    return fid

def remove_from_index(index: faiss.Index, faiss_id: int) -> None:
    sel = faiss.IDSelectorBatch(np.array([int(faiss_id)], dtype="int64"))
    index.remove_ids(sel)

def add_to_index(index: faiss.Index, faiss_id: int, vec_1xd: np.ndarray) -> None:
    index.add_with_ids(vec_1xd.astype("float32"), np.array([int(faiss_id)], dtype="int64"))

def bootstrap_ingest_staging_into_index(namespace: str, index: faiss.Index, state: Dict[str, Any]) -> None:
    """
    Move all staging vectors into IVF index (bulk-add), mark docs as in_index=1, delete staging rows.
    """
    staged = staging_all_vectors(namespace)
    if not staged:
        return

    # Bulk add in batches (laptop-friendly)
    B = 1000
    for i in range(0, len(staged), B):
        batch = staged[i:i+B]
        vecs = []
        ids = []
        for doc_id, vec in batch:
            # ensure doc has a faiss_id
            rec = meta_get_doc(namespace, doc_id)
            if not rec or rec["faiss_id"] is None:
                faiss_id = allocate_faiss_id(state)
            else:
                faiss_id = int(rec["faiss_id"])
            vecs.append(vec.reshape(1, -1))
            ids.append(faiss_id)

        X = np.vstack(vecs).astype("float32")
        I = np.array(ids, dtype="int64")
        index.add_with_ids(X, I)

        # update meta + delete staging
        for (doc_id, _), faiss_id in zip(batch, ids):
            rec = meta_get_doc(namespace, doc_id)
            if rec:
                meta_upsert_doc(
                    namespace,
                    doc_id,
                    faiss_id=faiss_id,
                    in_index=1,
                    doc_type=rec["type"],
                    text=rec.get("text", ""),
                    metadata=rec.get("metadata", {}),
                )
            staging_delete(namespace, doc_id, faiss_id=faiss_id, remove_from_mem=False)

    reset_staging_index(namespace, int(state.get("dim", DEFAULT_DIM)))
    save_shard_state(namespace, state)



# SEARCH (IVF + staging FlatIP)


def search_index(namespace: str, index: faiss.Index, q_1xd: np.ndarray, top_k: int, nprobe: int) -> List[Dict[str, Any]]:
    set_nprobe(index, nprobe)
    D, I = index.search(q_1xd.astype("float32"), max(1, top_k * OVERSAMPLE))
    ids = [int(x) for x in I[0].tolist() if int(x) != -1]
    if not ids:
        return []

    # fetch docs by faiss_id
    con = sqlite_connect(shard_meta_db(namespace))
    placeholders = ",".join(["?"] * len(ids))
    rows = con.execute(
        f"SELECT id, faiss_id, type, text, metadata_json, created_at, updated_at FROM docs WHERE faiss_id IN ({placeholders})",
        ids
    ).fetchall()
    con.close()

    by_faiss = {}
    for r in rows:
        by_faiss[int(r[1])] = {
            "id": r[0],
            "type": r[2],
            "text": r[3] or "",
            "metadata": json.loads(r[4] or "{}"),
            "created_at": r[5],
            "updated_at": r[6],
        }

    out = []
    for score, fid in zip(D[0].tolist(), I[0].tolist()):
        fid = int(fid)
        if fid == -1:
            continue
        rec = by_faiss.get(fid)
        if not rec:
            continue
        boost = 0.05 if rec.get("type") == "image" else 0.0
        out.append({
            "id": rec["id"],
            "score": float(score + boost),
            "type": rec["type"],
            "text": rec["text"],
            "metadata": rec["metadata"],
            "created_at": rec["created_at"],
            "updated_at": rec["updated_at"],
            "source": "ivf",
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]

def search_staging(namespace: str, q_1xd: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    index = get_or_build_staging_index(namespace, int(q_1xd.shape[-1]))
    if index.ntotal == 0:
        return []

    k = min(top_k, index.ntotal)
    D, I = index.search(q_1xd.astype("float32"), k)
    ids = [int(x) for x in I[0].tolist() if int(x) != -1]
    if not ids:
        return []

    con = sqlite_connect(shard_meta_db(namespace))
    placeholders = ",".join(["?"] * len(ids))
    rows = con.execute(
        f"SELECT id, faiss_id, type, text, metadata_json, created_at, updated_at FROM docs WHERE faiss_id IN ({placeholders})",
        ids
    ).fetchall()
    con.close()

    by_faiss = {}
    for r in rows:
        by_faiss[int(r[1])] = {
            "id": r[0],
            "type": r[2],
            "text": r[3] or "",
            "metadata": json.loads(r[4] or "{}"),
            "created_at": r[5],
            "updated_at": r[6],
        }

    out = []
    for score, fid in zip(D[0].tolist(), I[0].tolist()):
        fid = int(fid)
        if fid == -1:
            continue
        rec = by_faiss.get(fid)
        if not rec:
            continue
        boost = 0.05 if rec.get("type") == "image" else 0.0
        out.append({
            "id": rec["id"],
            "score": float(score + boost),
            "type": rec["type"],
            "text": rec["text"],
            "metadata": rec["metadata"],
            "created_at": rec["created_at"],
            "updated_at": rec["updated_at"],
            "source": "staging",
        })
    return out

def merge_results(a: List[Dict[str, Any]], b: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    merged = a + b
    # if same id appears twice, keep higher score
    best = {}
    for r in merged:
        rid = r["id"]
        if rid not in best or r["score"] > best[rid]["score"]:
            best[rid] = r
    out = list(best.values())
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]



# API


@app.get("/health")
def health():
    return jsonify({"ok": True, "shard_id": SHARD_ID})

@app.post("/insert")
def insert():
    namespace = None
    doc_id = None

    ct = (request.content_type or "").lower()

    # Lock per shard+namespace
    if "multipart/form-data" in ct:
        namespace = request.form.get("namespace", "default")
        doc_id = request.form.get("id")
        image = request.files.get("image")
        text = request.form.get("text")
        meta_raw = request.form.get("metadata")
        doc_meta = json.loads(meta_raw) if meta_raw else {}

        if not doc_id or not isinstance(doc_id, str):
            return jsonify({"error": "id is required (string)"}), 400
        if image is None and not text:
            return jsonify({"error": "provide image or text"}), 400

        init_shard_dbs(namespace)
        lock = FileLock(shard_lock_path(namespace))
        with lock:
            # make sure training pool exists
            init_shard_dbs(namespace)

            # if doc exists, remove from wherever it was
            existing = meta_get_doc(namespace, doc_id)

            # compute embedding
            if image is not None:
                keywords = extract_image_keywords(image)
                vec = clip_embed_image(image)
                doc_meta["filename"] = image.filename
                doc_meta["image_keywords"] = keywords
                doc_type = "image"
                doc_text = keywords
            else:
                vec = clip_embed_text(text or "")
                doc_type = "text"
                doc_text = text or ""

            # Add to training pool if template not trained yet
            pool_add_vector(namespace, f"{SHARD_ID}:{doc_id}", vec)

            # ensure namespace training if needed
            trained_now = run_namespace_training_if_needed(namespace, DEFAULT_DIM)

            index, state = ensure_shard_index_ready(namespace)

            # If we got trained, bootstrap ingest staged vectors into index
            if index is not None and trained_now:
                bootstrap_ingest_staging_into_index(namespace, index, state)
                save_faiss_index(index, shard_index_path(namespace))

            # Upsert logic
            if existing:
                # remove old
                if existing["in_index"] == 1 and index is not None and existing["faiss_id"] is not None:
                    remove_from_index(index, existing["faiss_id"])
                else:
                    staging_delete(namespace, doc_id, faiss_id=existing["faiss_id"])

            # If index trained and ready -> add directly
            if index is not None:
                faiss_id = existing["faiss_id"] if (existing and existing["faiss_id"] is not None) else allocate_faiss_id(state)
                add_to_index(index, faiss_id, vec)
                meta_upsert_doc(namespace, doc_id, faiss_id, 1, doc_type, doc_text, doc_meta)
                save_faiss_index(index, shard_index_path(namespace))
                save_shard_state(namespace, state)
                return jsonify({"namespace": namespace, "result": {"id": doc_id, "faiss_id": int(faiss_id), "stored": "ivf"}})

            # else -> staging
            # allocate faiss_id early so it stays stable after training
            faiss_id = existing["faiss_id"] if (existing and existing["faiss_id"] is not None) else allocate_faiss_id(state)
            staging_put(namespace, doc_id, vec, faiss_id=faiss_id)
            meta_upsert_doc(namespace, doc_id, faiss_id, 0, doc_type, doc_text, doc_meta)
            save_shard_state(namespace, state)
            return jsonify({"namespace": namespace, "result": {"id": doc_id, "faiss_id": int(faiss_id), "stored": "staging"}})

    # JSON (text-only)
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    doc_id = body.get("id")
    text = body.get("text", "")
    doc_meta = body.get("metadata", {}) or {}

    if not doc_id or not isinstance(doc_id, str):
        return jsonify({"error": "id is required (string)"}), 400
    if not text or not isinstance(text, str):
        return jsonify({"error": "text is required (string)"}), 400

    init_shard_dbs(namespace)
    lock = FileLock(shard_lock_path(namespace))
    with lock:
        existing = meta_get_doc(namespace, doc_id)

        vec = clip_embed_text(text)
        doc_type = "text"
        doc_text = text

        pool_add_vector(namespace, f"{SHARD_ID}:{doc_id}", vec)
        trained_now = run_namespace_training_if_needed(namespace, DEFAULT_DIM)

        index, state = ensure_shard_index_ready(namespace)

        if index is not None and trained_now:
            bootstrap_ingest_staging_into_index(namespace, index, state)
            save_faiss_index(index, shard_index_path(namespace))

        if existing:
            if existing["in_index"] == 1 and index is not None and existing["faiss_id"] is not None:
                remove_from_index(index, existing["faiss_id"])
            else:
                staging_delete(namespace, doc_id, faiss_id=existing["faiss_id"])

        if index is not None:
            faiss_id = existing["faiss_id"] if (existing and existing["faiss_id"] is not None) else allocate_faiss_id(state)
            add_to_index(index, faiss_id, vec)
            meta_upsert_doc(namespace, doc_id, faiss_id, 1, doc_type, doc_text, doc_meta)
            save_faiss_index(index, shard_index_path(namespace))
            save_shard_state(namespace, state)
            return jsonify({"namespace": namespace, "result": {"id": doc_id, "faiss_id": int(faiss_id), "stored": "ivf"}})

        faiss_id = existing["faiss_id"] if (existing and existing["faiss_id"] is not None) else allocate_faiss_id(state)
        staging_put(namespace, doc_id, vec, faiss_id=faiss_id)
        meta_upsert_doc(namespace, doc_id, faiss_id, 0, doc_type, doc_text, doc_meta)
        save_shard_state(namespace, state)
        return jsonify({"namespace": namespace, "result": {"id": doc_id, "faiss_id": int(faiss_id), "stored": "staging"}})


@app.post("/search")
def search():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    query = body.get("query", "")
    top_k = int(body.get("top_k", DEFAULT_TOP_K))
    nprobe = int(body.get("nprobe", DEFAULT_NPROBE))

    if not query:
        return jsonify({"error": "query required"}), 400

    init_shard_dbs(namespace)
    lock = FileLock(shard_lock_path(namespace))
    with lock:
        index, state = ensure_shard_index_ready(namespace)
        q = clip_embed_text(query)

        res_ivf = search_index(namespace, index, q, top_k, nprobe) if index is not None else []
        res_stg = search_staging(namespace, q, top_k)
        merged = merge_results(res_ivf, res_stg, top_k)

        return jsonify({
            "namespace": namespace,
            "shard_id": SHARD_ID,
            "trained": bool(index is not None),
            "results": merged
        })


@app.post("/search/image")
def search_by_image():
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    namespace = request.form.get("namespace", "default")
    top_k = int(request.form.get("top_k", DEFAULT_TOP_K))
    nprobe = int(request.form.get("nprobe", DEFAULT_NPROBE))

    init_shard_dbs(namespace)
    lock = FileLock(shard_lock_path(namespace))
    with lock:
        index, state = ensure_shard_index_ready(namespace)
        q = clip_embed_image(request.files["image"])

        res_ivf = search_index(namespace, index, q, top_k, nprobe) if index is not None else []
        res_stg = search_staging(namespace, q, top_k)
        merged = merge_results(res_ivf, res_stg, top_k)

        return jsonify({
            "namespace": namespace,
            "shard_id": SHARD_ID,
            "trained": bool(index is not None),
            "results": merged
        })
    
@app.post("/admin/retrain")
def retrain():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    sample_last = body.get("last_n")  # optional

    ok = run_namespace_training_if_needed(
        namespace,
        DEFAULT_DIM,
        force=True,
        sample_limit=sample_last
    )

    if not ok:
        return jsonify({"error": "Not enough data to retrain"}), 400

    return jsonify({
        "namespace": namespace,
        "status": "retrained",
        "used_vectors": sample_last or "auto"
    })

@app.post("/admin/namespace/create")
def create_namespace():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    if not namespace or not isinstance(namespace, str):
        return jsonify({"error": "namespace is required (string)"}), 400

    init_shard_dbs(namespace)
    ensure_dir(ns_training_dir(namespace))
    state = load_or_create_shard_state(namespace)
    save_shard_state(namespace, state)

    return jsonify({
        "namespace": namespace,
        "status": "created",
        "shard_id": SHARD_ID,
    })

@app.get("/admin/namespaces")
def get_namespaces():
    return jsonify({
        "shard_id": SHARD_ID,
        "namespaces": list_namespaces(),
    })

@app.get("/admin/namespace")
def get_namespace():
    namespace = request.args.get("namespace", "default")
    if not namespace:
        return jsonify({"error": "namespace is required"}), 400

    init_shard_dbs(namespace)
    total, in_index = count_meta_docs(namespace)
    staged = count_staging_vectors(namespace)
    trained_template = is_trained_template_exists(namespace)
    index_exists = os.path.exists(shard_index_path(namespace))
    include_training_stats = parse_bool(request.args.get("include_training_stats"))
    stats = read_json(ns_training_stats(namespace), default=None) if include_training_stats else None

    return jsonify({
        "namespace": namespace,
        "shard_id": SHARD_ID,
        "counts": {
            "docs_total": total,
            "docs_in_index": in_index,
            "staging_vectors": staged,
            "training_pool": pool_count(namespace),
        },
        "index": {
            "trained_template": trained_template,
            "index_exists": index_exists,
        },
        "training_stats": stats,
    })

@app.get("/admin/namespace/docs")
def list_namespace_docs():
    namespace = request.args.get("namespace", "default")
    if not namespace:
        return jsonify({"error": "namespace is required"}), 400

    limit = int(request.args.get("limit", "50"))
    limit = max(1, min(limit, 500))
    after_id = request.args.get("after_id")
    include_text = parse_bool(request.args.get("include_text"))
    include_metadata = parse_bool(request.args.get("include_metadata"))
    include_total = parse_bool(request.args.get("include_total"))

    init_shard_dbs(namespace)
    con = sqlite_connect(shard_meta_db(namespace))
    if after_id:
        rows = con.execute(
            """
            SELECT id, faiss_id, in_index, type, text, metadata_json, created_at, updated_at
            FROM docs
            WHERE id > ?
            ORDER BY id
            LIMIT ?
            """,
            (after_id, limit)
        ).fetchall()
    else:
        rows = con.execute(
            """
            SELECT id, faiss_id, in_index, type, text, metadata_json, created_at, updated_at
            FROM docs
            ORDER BY id
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

    total = None
    if include_total:
        total = int(con.execute("SELECT COUNT(*) FROM docs;").fetchone()[0])
    con.close()

    docs = []
    last_id = None
    for r in rows:
        doc = {
            "id": r[0],
            "faiss_id": int(r[1]) if r[1] is not None else None,
            "in_index": int(r[2]),
            "type": r[3],
            "created_at": r[6],
            "updated_at": r[7],
        }
        if include_text:
            doc["text"] = r[4] or ""
        if include_metadata:
            doc["metadata"] = json.loads(r[5] or "{}")
        docs.append(doc)
        last_id = r[0]

    return jsonify({
        "namespace": namespace,
        "shard_id": SHARD_ID,
        "docs": docs,
        "next_after_id": last_id if len(docs) == limit else None,
        "total": total,
    })

@app.delete("/delete")
def delete_doc():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "default")
    doc_id = body.get("id")
    if not doc_id or not isinstance(doc_id, str):
        return jsonify({"error": "id is required (string)"}), 400

    init_shard_dbs(namespace)
    lock = FileLock(shard_lock_path(namespace))
    with lock:
        rec = meta_get_doc(namespace, doc_id)
        if not rec:
            return jsonify({"namespace": namespace, "id": doc_id, "status": "not_found"}), 404

        index, state = ensure_shard_index_ready(namespace)
        if rec["in_index"] == 1:
            if index is None or rec["faiss_id"] is None:
                return jsonify({"error": "index not ready for delete"}), 500
            remove_from_index(index, rec["faiss_id"])
            save_faiss_index(index, shard_index_path(namespace))
        else:
            staging_delete(namespace, doc_id, faiss_id=rec.get("faiss_id"))

        meta_delete_doc(namespace, doc_id)
        pool_delete_vector(namespace, f"{SHARD_ID}:{doc_id}")

        return jsonify({
            "namespace": namespace,
            "id": doc_id,
            "status": "deleted",
            "in_index": rec["in_index"],
        })

@app.delete("/admin/namespace")
def delete_namespace():
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "")
    if not namespace or not isinstance(namespace, str):
        return jsonify({"error": "namespace is required (string)"}), 400

    base = os.path.abspath(ns_base(namespace))
    ns_root = os.path.abspath(NS_DIR)
    if not base.startswith(ns_root + os.sep):
        return jsonify({"error": "invalid namespace"}), 400

    if namespace in STAGING_FLAT_INDEXES:
        del STAGING_FLAT_INDEXES[namespace]

    if not os.path.exists(base):
        return jsonify({
            "namespace": namespace,
            "status": "not_found",
            "deleted": False,
            "shard_id": SHARD_ID,
        })

    try:
        shutil.rmtree(base)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "namespace": namespace,
        "status": "deleted",
        "deleted": True,
        "shard_id": SHARD_ID,
    })



if __name__ == "__main__":
    ensure_dir(NS_DIR)
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
