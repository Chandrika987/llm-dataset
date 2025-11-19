# clean_jsonl.py
import json, re, sys
from pathlib import Path

URL_RE = re.compile(r"https?://\S+")
HTML_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(s, lower=True):
    if s is None:
        return s
    s = URL_RE.sub("", s)
    s = HTML_RE.sub("", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    if lower:
        s = s.lower()
    return s

def load_jsonl(path):
    items=[]
    with open(path, "r", encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            line=line.strip()
            if not line: 
                continue
            try:
                obj=json.loads(line)
                items.append(obj)
            except Exception as e:
                print(f"[WARN] line {i} not valid JSON: {e}")
    return items

def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main(infile, outfile_clean, outfile_pretty):
    items = load_jsonl(infile)
    print(f"loaded {len(items)} items")
    seen=set()
    out=[]
    for obj in items:
        # Basic safety: ensure fields exist
        title = obj.get("title") or ""
        text  = obj.get("text")  or ""
        # Normalize for dedupe & cleaning
        norm_title = normalize_text(title, lower=True)
        norm_text  = normalize_text(text, lower=True)
        key = (norm_title, norm_text)
        if key in seen:
            continue
        seen.add(key)
        # Update object with cleaned text (but keep original fields too if you want)
        obj["text_clean"] = norm_text
        obj["title_clean"] = norm_title
        # Add helpful metadata if missing
        if "domain" not in obj:
            obj["domain"] = "agriculture"
        out.append(obj)
    print(f"after dedupe: {len(out)} items")
    save_jsonl(out, outfile_clean)
    # save pretty JSON array too for easy inspection
    with open(outfile_pretty, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved", outfile_clean, outfile_pretty)

if __name__=="__main__":
    if len(sys.argv) != 4:
        print("Usage: python clean_jsonl.py input.jsonl output_clean.jsonl output_pretty.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
