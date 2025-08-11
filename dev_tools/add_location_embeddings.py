"""Add location embeddings from locations_folders.csv into the existing ChromaDB vector store.

This script reads a CSV file that contains timestamped location information and
stores each row as a vector embedding in the existing `ofergpt_memories`
collection. It utilises the existing `RAGSystem` infrastructure so that all
settings (Cohere embeddings, Chroma persistent client, text splitter, etc.) are
kept consistent with the rest of the project.

Usage:
    python scripts/add_location_embeddings.py --csv locations_folders.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any

from langchain.schema import Document

from ..rag_system import RAGSystem

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_documents(rows: List[Dict[str, str]]) -> List[Document]:
    """Convert CSV rows into LangChain `Document` objects suitable for ChromaDB."""
    documents: List[Document] = []

    for row in rows:
        timestamp = row.get("Timestamp", "").strip()
        folder = row.get("Foldername", "").strip()
        city = row.get("City", "").strip()
        country = row.get("Country", "").strip()

        # Skip completely empty rows
        if not any([timestamp, folder, city, country]):
            continue

        # Build human-readable content string for embedding
        location_parts = [p for p in (city, country) if p]
        location_str = ", ".join(location_parts) if location_parts else "Unknown location"
        content = f"Visit to {location_str} (folder: {folder}) on {timestamp}."

        metadata: Dict[str, Any] = {
            "type": "location",
            "timestamp": timestamp,
            "folder": folder,
            "city": city,
            "country": country,
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents

# ---------------------------------------------------------------------------
# Embedding logic (callable)
# ---------------------------------------------------------------------------

def embed_locations(csv_path: str | Path = "locations_folders.csv") -> None:
    """Read a CSV file and add its locations as embeddings to ChromaDB.

    Safe to call multiple times – duplicates are skipped automatically.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV rows
    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    if not rows:
        print("No rows found in CSV – nothing to embed.")
        return

    rag = RAGSystem()

    # Build documents from CSV
    docs = build_documents(rows)

    # Gather existing location signatures from the vector store to avoid re-adding
    existing_signatures: set[tuple[str, str, str, str]] = set()
    try:
        collection = rag.chroma_client.get_or_create_collection("ofergpt_memories")
        data = collection.get()
        if data and data.get("metadatas"):
            for meta in data["metadatas"]:
                if not meta:
                    continue
                if meta.get("type") == "location":
                    sig = (
                        meta.get("timestamp", ""),
                        meta.get("folder", ""),
                        meta.get("city", ""),
                        meta.get("country", ""),
                    )
                    existing_signatures.add(sig)
    except Exception as e:
        print(f"[LOC] Could not list existing locations for dedupe: {e}")

    # De-duplicate within the current batch AND against existing signatures
    batch_signatures: set[tuple[str, str, str, str]] = set()
    unique_docs: List[Document] = []
    for doc in docs:
        meta = doc.metadata
        signature = (
            meta.get("timestamp", ""),
            meta.get("folder", ""),
            meta.get("city", ""),
            meta.get("country", ""),
        )
        if signature in batch_signatures or signature in existing_signatures:
            continue
        batch_signatures.add(signature)
        unique_docs.append(doc)

    if not unique_docs:
        print("All location documents already embedded – nothing to add.")
        return

    # Split and add to vector store via RAGSystem utilities
    split_docs = rag.text_splitter.split_documents(unique_docs)
    rag.vectorstore.add_documents(split_docs)

    print(f"✅ Added {len(split_docs)} location document chunks to vector store ({len(unique_docs)} new locations)")
    # Refresh CSV dump for visibility
    try:
        rag._dump_embeddings_csv()
    except Exception as e:
        print(f"[CSV] Could not update embeddings dump after locations: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create embeddings for location CSV entries")
    parser.add_argument("--csv", type=str, default="locations_folders.csv", help="Path to CSV file with location data")
    args = parser.parse_args()
    embed_locations(args.csv)


if __name__ == "__main__":
    main()
