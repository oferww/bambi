"""Utility: dump stored documents and metadata from ChromaDB.

Usage (inside Docker container or host with correct path):

    python scripts/dump_embeddings.py --limit 100
    python scripts/dump_embeddings.py --where '{"source":"photo"}' --out photos.csv

If no --out is given, prints to console.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from chromadb import PersistentClient

DEFAULT_DB_PATH = Path("data/embeddings")
DEFAULT_COLLECTION = "ofergpt_memories"


def parse_where(condition: str | None) -> Dict[str, Any] | None:
    if not condition:
        return None
    try:
        return json.loads(condition)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON for --where: {e}")


def fetch_docs(limit: int, where: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    client = PersistentClient(str(DEFAULT_DB_PATH))
    coll = client.get_or_create_collection(DEFAULT_COLLECTION)

    get_kwargs = {"include": ["documents", "metadatas"]}
    if where:
        get_kwargs["where"] = where
    data = coll.get(**get_kwargs)

    rows = []
    for idx, doc in enumerate(data["documents"]):
        if limit and idx >= limit:
            break
        meta = data["metadatas"][idx]
        rows.append({
            "id": data["ids"][idx],
            "source": meta.get("source", ""),
            "location": meta.get("location_name", ""),
            "timestamp": meta.get("timestamp") or meta.get("date_taken", ""),
            "text": doc,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Dump Bambi embeddings store to console or CSV")
    ap.add_argument("--limit", type=int, default=50, help="Max rows to output (default 50, 0 = all)")
    ap.add_argument("--where", type=str, help="JSON metadata filter (e.g. '{\"source\":\"photo\"}')")
    ap.add_argument("--contains", type=str, help="Case-insensitive substring filter on document text")
    ap.add_argument("--out", type=Path, help="Write CSV to this path instead of stdout")
    args = ap.parse_args()

    rows = fetch_docs(args.limit, parse_where(args.where))

    if args.contains:
        rows = [r for r in rows if args.contains.lower() in r["text"].lower()]

    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys()) if rows else csv.writer(f)
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"✅ Wrote {len(rows)} rows to {args.out}")
    else:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
