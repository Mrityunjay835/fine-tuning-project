import json

objects = []
buffer = ""
depth = 0

with open("data/datasets._rawjson", "r", encoding="utf-8") as f:
    text = f.read()

# Fix common bad escapes
text = text.replace("\\'", "'").replace("\\–", "–").replace("\\₹", "₹")

for ch in text:
    if ch == "{":
        depth += 1
    if depth > 0:
        buffer += ch
    if ch == "}":
        depth -= 1
        if depth == 0:
            try:
                objects.append(json.loads(buffer))
            except Exception as e:
                print("Skipping bad record:", e)
            buffer = ""

with open("data/datasets.json", "w", encoding="utf-8") as f:
    for obj in objects:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Converted {len(objects)} records successfully.")
