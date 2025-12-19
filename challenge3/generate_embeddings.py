import json
import random
import numpy as np
from pathlib import Path

# ---------------- Safe Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "cleaned" / "agri_dataset_clean.jsonl"
OUTPUT_PATH = BASE_DIR / "agri_embeddings.txt"


# ---------------- Simple Tokenizer ----------------
# (uses whitespace, since your BPE tokenizer is separate)
def simple_tokenize(text):
    return text.lower().split()

# ---------------- Load Dataset ----------------
texts = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])

print(f"Loaded {len(texts)} text samples")

# ---------------- Build Vocabulary ----------------
vocab = {}
tokenized_texts = []

for text in texts:
    tokens = simple_tokenize(text)
    tokenized_texts.append(tokens)
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

vocab_size = len(vocab)
embedding_dim = 128

print(f"Vocabulary size: {vocab_size}")

# ---------------- Initialize Embeddings ----------------
# Random initialization (like Word2Vec start)
embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)

# ---------------- Save Output ----------------
np.save(OUTPUT_PATH, embeddings)

print("Embeddings generated successfully")
print("Saved to:", OUTPUT_PATH)
