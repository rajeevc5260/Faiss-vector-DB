import requests
import textwrap
import uuid

API_URL = "http://localhost:5001/insert"  # router
NAMESPACE = "data"
CHUNK_SIZE = 100  # characters per chunk
INPUT_FILE = "input.txt"

# Read text from file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks
chunks = textwrap.wrap(text, CHUNK_SIZE)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    payload = {
        "namespace": NAMESPACE,
        "id": f"doc_{uuid.uuid4().hex}",
        "text": chunk,
        "metadata": {
            "chunk_index": i
        }
    }

    r = requests.post(API_URL, json=payload)
    if not r.ok:
        print(f"[{i}] failed → {r.text}")
        break

    stored = r.json()["response"]["result"]["stored"]
    print(f"[{i}] inserted → {stored}")
