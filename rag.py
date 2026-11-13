import os
import re

import numpy as np
import pymupdf
from rank_bm25 import BM25Okapi

MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
_model = None
_bm25_cache = {}


def get_key():
    if os.path.exists("key.txt"):
        return open("key.txt").read().strip()
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = ""
    return key


def load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model


def _tokens(text):
    return re.findall(r"\w+", text.lower())


def _get_bm25(chunks):
    key = id(chunks)
    if key not in _bm25_cache:
        _bm25_cache.clear()
        _bm25_cache[key] = BM25Okapi([_tokens(c["text"]) for c in chunks])
    return _bm25_cache[key]


def build_index(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        start = 0
        while start < len(text):
            piece = text[start:start + 800]
            if len(piece.strip()) > 100:
                chunks.append({"text": piece, "page": page_number})
            start += 650
    embs = load_model().encode([c["text"] for c in chunks], normalize_embeddings=True)
    return chunks, embs


def search(question, chunks, embs, k=6, pool=30, rrf_k=60):
    """Hybrid search: meaning (embeddings) + exact words (BM25), merged by rank."""
    q_emb = load_model().encode([question], normalize_embeddings=True)
    by_meaning = np.argsort(np.dot(embs, q_emb[0]))[::-1][:pool]
    word_scores = _get_bm25(chunks).get_scores(_tokens(question))
    by_words = [i for i in np.argsort(word_scores)[::-1][:pool] if word_scores[i] > 0]

    score = {}
    for rank, i in enumerate(by_meaning):
        score[int(i)] = score.get(int(i), 0) + 1 / (rrf_k + rank)
    for rank, i in enumerate(by_words):
        score[int(i)] = score.get(int(i), 0) + 1 / (rrf_k + rank)
    return sorted(score, key=score.get, reverse=True)[:k]


def ask(question, chunks, embs, client, k=12):
    top = search(question, chunks, embs, k=k)
    context = "".join(f"[Page {chunks[i]['page']}]\n{chunks[i]['text']}\n\n" for i in top)
    prompt = f"""Answer ONLY using the context below.
Mention the page number like (Page 3) after each fact.
If the answer is not in the context, say: "I could not find this in the document."

Context:
{context}

Question: {question}"""
    answer = None
    for name in MODELS:
        try:
            answer = client.models.generate_content(model=name, contents=prompt).text
            break
        except Exception:
            continue
    if answer is None:
        answer = "Google is busy right now. Please try again in a few minutes."
    return answer, top