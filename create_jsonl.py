import json

input_file = "agri_dataset_100.jsonl"
output_file = "agri_dataset_clean.jsonl"

cleaned = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue  # skip empty lines
        
        try:
            item = json.loads(line)
        except:
            continue  # skip broken JSON lines

        # Clean content
        if "content" in item:
            item["content"] = item["content"].replace("\n", " ").strip()

        cleaned.append(item)

with open(output_file, "w", encoding="utf-8") as f:
    for c in cleaned:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print("✔ Cleaning complete! Saved as agri_dataset_clean.jsonl")
