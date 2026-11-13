import time

import pandas as pd
from google import genai

import rag

PDF_FILE = "thinkpython.pdf"      # the book you downloaded
QUESTIONS_FILE = "questions.csv"

client = genai.Client(api_key=rag.get_key())
chunks, embs = rag.build_index(open(PDF_FILE, "rb").read())
tests = pd.read_csv(QUESTIONS_FILE, dtype=str).fillna("")


def has_any(options, text):
    """True if any of the options (separated by |) appears in the text."""
    return any(o.strip().lower() in text for o in options.split("|") if o.strip())


rows = []
found_ok = found_n = refuse_ok = refuse_n = 0

for _, t in tests.iterrows():
    answer, top = rag.ask(t["question"], chunks, embs, client)
    if answer.startswith("Google is busy"):
        print("SKIPPED (Google busy):", t["question"])
        continue
    low = answer.lower()
    refused = "could not find" in low

    if t["keyword"].strip() == "":
        refuse_n += 1
        ok = refused
        refuse_ok += ok
        kind = "not in PDF"
    else:
        found_n += 1
        ok = has_any(t["keyword"], low) and not refused
        found_ok += ok
        kind = "in PDF"

    print(f"{'PASS' if ok else 'FAIL'} | {kind} | {t['question']}")
    rows.append({"question": t["question"], "type": kind, "pass": ok, "answer": answer})
    time.sleep(6)

pd.DataFrame(rows).to_csv("eval_pdf_results.csv", index=False)
print()
print(f"Questions in the book: answer correct {found_ok} of {found_n}")
print(f"Questions not in the book: correctly refused {refuse_ok} of {refuse_n}")