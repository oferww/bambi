"""Embed JSON/NDJSON files from a folder into the local ChromaDB store.

Usage:
    # Default folder: ./data/uploads/json
    python -m dev_tools.add_json_embeddings

    # Custom folder
    python -m dev_tools.add_json_embeddings --dir ./data/custom_json

This uses the same ingestion logic as the Streamlit app, routing:
- Instagram exports to the Instagram-specific handler
- Other JSONs to a generic JSON -> Document pipeline

All vectors are stored locally using RAGSystem's Chroma PersistentClient
at ./data/embeddings (no S3).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from backend.rag_system import RAGSystem
# We intentionally import the internal helper; it is stable within this repo
from backend.ingestion.dispatcher import _ingest_generic_json

# Load environment variables from .env for local runs
load_dotenv()


def embed_jsons(dir_path: str | Path = "./data/uploads/json") -> int:
    """Embed all .json/.jsonl files found under dir_path (non-recursive).

    Returns the total number of documents/posts ingested.
    """
    p = Path(dir_path)
    os.makedirs(p, exist_ok=True)

    rag = RAGSystem()

    total = 0
    try:
        for fname in sorted(os.listdir(p)):
            if not fname.lower().endswith((".json", ".jsonl")):
                continue
            fpath = p / fname
            try:
                total += _ingest_generic_json(str(fpath), rag)
            except Exception as e:
                print(f"[JSON-EMBED] Error processing '{fname}': {e}")
    except Exception as e:
        print(f"[JSON-EMBED] Error scanning folder '{p}': {e}")

    # Best-effort summary
    print(f"✅ Embedded JSONs from {p} (added {total} docs/posts)")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed JSON/NDJSON files from a folder into local vector store")
    ap.add_argument("--dir", dest="dir", default="./data/uploads/json", help="Folder containing .json/.jsonl files")
    args = ap.parse_args()

    embed_jsons(args.dir)


if __name__ == "__main__":
    main()
