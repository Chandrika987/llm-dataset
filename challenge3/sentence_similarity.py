from pathlib import Path
import math

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_FILE = BASE_DIR / "agri_embeddings.txt"

print("Looking for embeddings at:", EMBEDDING_FILE)


# ---------------- Load Embeddings ----------------
embeddings = {}

with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vector = list(map(float, parts[1:]))
        embeddings[word] = vector

# ---------------- Helper Functions ----------------
def tokenize(text):
    return text.lower().split()

def average_embedding(tokens):
    vectors = [embeddings[t] for t in tokens if t in embeddings]
    if not vectors:
        return None
    return [sum(col) / len(vectors) for col in zip(*vectors)]

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot / (mag1 * mag2)

# ---------------- Sample Sentences ----------------
sentence_1 = "soil moisture improves crop yield"
sentence_2 = "irrigation increases agricultural productivity"
sentence_3 = "weather affects daily routine"

# ---------------- Compute Similarity ----------------
vec1 = average_embedding(tokenize(sentence_1))
vec2 = average_embedding(tokenize(sentence_2))
vec3 = average_embedding(tokenize(sentence_3))

sim_1_2 = cosine_similarity(vec1, vec2)
sim_1_3 = cosine_similarity(vec1, vec3)

print("Sentence Similarity Results")
print("----------------------------")
print(f"Similarity (S1 vs S2): {sim_1_2:.3f}")
print(f"Similarity (S1 vs S3): {sim_1_3:.3f}")
