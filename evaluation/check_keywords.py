import json, re, sys
from pathlib import Path
from pypdf import PdfReader

DOCS = Path("data/sample_docs")
ITEMS = json.loads(Path("evaluation/eval_set.json").read_text(encoding="utf-8"))

def norm(s):
    s = s.lower().replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", s)

docs = {}  # name -> list of (page_number, text)
for f in DOCS.iterdir():
    if f.suffix.lower() == ".pdf":
        pages = [(i + 1, norm(p.extract_text() or "")) for i, p in enumerate(PdfReader(f).pages)]
    elif f.suffix.lower() in (".txt", ".md"):
        pages = [(None, norm(f.read_text(encoding="utf-8", errors="ignore")))]
    else:
        continue
    docs[f.name] = pages

for it in ITEMS:
    match = [n for n in docs if it["expected_source"].lower() in n.lower()]
    if not match:
        print(f"Q{it['id']}: NO FILE matches expected_source '{it['expected_source']}'")
        continue
    pages = docs[match[0]]
    full = " ".join(t for _, t in pages)
    missing = [k for k in it["expected_keywords"] if norm(k) not in full]
    found_pages = sorted({p for p, t in pages if p and all(norm(k) in t for k in it["expected_keywords"])})
    status = "OK " if not missing else "FIX"
    extra = f" | page in file: {it.get('source_page')} | pages with keywords: {found_pages}" if pages[0][0] else ""
    print(f"{status} Q{it['id']}: missing={missing}{extra}")