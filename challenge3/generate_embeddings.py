import json
import random
from pathlib import Path

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "cleaned" / "agri_dataset_clean.jsonl"
OUT_FILE = BASE_DIR / "agri_embeddings.txt"

EMBEDDING_DIM = 8
random.seed(42)

# ---------------- Load Text ----------------
texts = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])

print(f"Loaded {len(texts)} text samples")

# ---------------- Build Vocabulary ----------------
vocab = set()
for text in texts:
    for word in text.lower().split():
        vocab.add(word)

print(f"Vocabulary size: {len(vocab)}")

# ---------------- Generate Random Embeddings ----------------
embeddings = {}

for word in vocab:
    embeddings[word] = [
        round(random.uniform(-1, 1), 4)
        for _ in range(EMBEDDING_DIM)
    ]

# ---------------- Save as TEXT ----------------
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for word, vector in embeddings.items():
        f.write(word + " " + " ".join(map(str, vector)) + "\n")

print("Embeddings generated successfully")
print(f"Saved to: {OUT_FILE}")
