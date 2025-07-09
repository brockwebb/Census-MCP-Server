#!/usr/bin/env python
"""
build_stats_faiss.py — create a FAISS index for fast similarity search
--------------------------------------------------------------------
• Encodes every ACS‑variable description with the chosen SBERT/BGE model.
• Accepts any universe file format:
  1. JSON‑Lines (one object per line)
  2. Classic JSON list   → [ {..}, {..}, … ]
  3. Classic JSON dict   → { "variables": [ … ] }
• Writes two artefacts to <out_dir>/
    ├─ variables_bge.faiss   (vector index)
    └─ variables_meta.json   (row count + metadata)

Usage
-----
    python build_stats_faiss.py \
        --universe  knowledge-base/2023_ACS_Enriched_Universe.json \
        --model     BAAI/bge-large-en-v1.5 \
        --out-dir   stats-index
"""

import argparse, json, pathlib, sys
from typing import List, Dict, Any

import faiss                    # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_universe(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Return a list of variable records regardless of file format."""

    with path.open("r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
        fh.seek(0)

        # --- JSON‑Lines ------------------------------------------------------
        if first_line.startswith("{") and first_line.endswith("}"):
            return [json.loads(line) for line in fh if line.strip()]

        # --- Classic JSON ----------------------------------------------------
        doc = json.load(fh)
        if isinstance(doc, list):
            return doc
        if isinstance(doc, dict) and "variables" in doc:
            return doc["variables"]

    raise ValueError("Unsupported universe format — cannot parse.")


def build_text(row: Any) -> str:
    """Return concatenated text for embedding (robust to non‑dict rows)."""
    if not isinstance(row, dict):
        return str(row)
    parts = [row.get("label", ""), row.get("concept", ""), row.get("enrichment_text", "")]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for ACS variables")
    parser.add_argument("--universe", required=True, help="Path to enriched universe JSON")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="Sentence‑Transformer model")
    parser.add_argument("--out-dir", default="stats-index", help="Directory to write index + meta")
    args = parser.parse_args(argv)

    uni_path = pathlib.Path(args.universe).expanduser()
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    if not uni_path.exists():
        sys.exit(f"❌ Universe file not found: {uni_path}")

    rows = load_universe(uni_path)
    print(f"▶ Loaded {len(rows):,} variable records → embedding with {args.model}")

    model = SentenceTransformer(args.model)
    texts = [build_text(r) for r in rows]

    vecs = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, str(out_dir / "variables_bge.faiss"))
    meta = {"rows": len(rows), "dim": vecs.shape[1], "model": args.model}
    json.dump(meta, open(out_dir / "variables_meta.json", "w"), indent=2)

    print("✅ FAISS index written →", out_dir / "variables_bge.faiss")


if __name__ == "__main__":
    main()
