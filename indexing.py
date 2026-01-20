import os
import json
from collections import defaultdict

BASE_DIR = os.path.expanduser("~")
INPUT_FOLDER = os.path.join(BASE_DIR, "Desktop", "clean_articles")
OUTPUT_FILE = os.path.join(BASE_DIR, "Desktop", "inverted_index.json")

inverted_index = defaultdict(list)

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".txt"):
        with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as f:
            tokens = f.read().split()

        for token in set(tokens):
            inverted_index[token].append(filename)

# Sort postings
for token in inverted_index:
    inverted_index[token] = sorted(inverted_index[token])

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=2)

print(f"✅ Inverted index saved at {OUTPUT_FILE}")
