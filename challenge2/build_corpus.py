import json
import os

# Absolute path to dataset
input_file = r"C:\Users\chand\OneDrive\Desktop\llm\data\cleaned\agri_dataset_clean.jsonl"

# Output corpus file inside challenge2
output_file = r"C:\Users\chand\OneDrive\Desktop\llm\challenge2\corpus.txt"

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        outfile.write(line)


print(f"Corpus file created: {output_file}")
