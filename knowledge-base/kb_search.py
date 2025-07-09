# kb_search.py – minimal helper to query the stats FAISS index
# ------------------------------------------------------------------
# Usage:
#   from kb_search import search
#   hits = search("median household income", k=8)
#   for h in hits: print(h["score"], h["source"])

import json
import pathlib
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# Config — adjust paths if you move the files ------------------------
# -------------------------------------------------------------------
BASE_DIR   = pathlib.Path(__file__).resolve().parent
INDEX_FP   = BASE_DIR / "stats-index/variables_bge.faiss"
META_FP    = BASE_DIR / "stats-index/variables_meta.json"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# -------------------------------------------------------------------
# Lazy load heavy objects -------------------------------------------
# -------------------------------------------------------------------
_index = None
_meta  = None
_model = None


def _load():
    global _index, _meta, _model
    if _index is None:
        if not INDEX_FP.exists():
            raise FileNotFoundError(f"FAISS index not found: {INDEX_FP}")
        _index = faiss.read_index(str(INDEX_FP))
    if _meta is None:
        linked = INDEX_FP.with_name("variables_meta.json")
        fp = linked if linked.exists() else META_FP
        _meta = json.load(open(fp))
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)


def search(query: str, k: int = 8) -> List[Dict]:
    """Return top‑k variable hits with similarity scores in [0‑1]."""
    _load()
    # encode & normalise
    qv = _model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = _index.search(qv, k)  # cosine sim because vectors are unit‑norm
    hits = []
    for idx, dist in zip(I[0], D[0]):
        meta = _meta[str(idx)] if isinstance(_meta, dict) else _meta[idx]
        hits.append({
            "score": float(dist),  # already cosine similarity in [‑1,1]; should be ≥0
            **meta
        })
    return hits

