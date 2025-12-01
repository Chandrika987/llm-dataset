import json
import random
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

# Path to your dataset
DATASET = r"C:\Users\chand\OneDrive\Desktop\llm\data\cleaned\agri_dataset_clean.jsonl"

# Read all lines
lines = []
with open(DATASET, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        # change 'text' to whatever key your dataset uses
        text = data.get("text") or data.get("content") or data.get("sentence")
        if text:
            lines.append(text)

# Pick 5 random samples
samples = random.sample(lines, 5)

print("\n=== Testing tokenizer on real dataset sentences ===\n")

for text in samples:
    print("Original:", text)

    encoded = tokenizer.encode(text)
    print("Token IDs:", encoded.ids)
    print("Tokens:", encoded.tokens)

    decoded = tokenizer.decode(encoded.ids)
    print("Decoded:", decoded)

    print("Match:", "YES" if decoded == text else "NO")
    print("-" * 70)
