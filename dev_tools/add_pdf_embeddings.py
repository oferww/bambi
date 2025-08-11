"""Add PDF document embeddings from a folder on startup.

Usage (CLI):
    python scripts/add_pdf_embeddings.py --dir ./data/uploads/pdfs

Programmatic usage:
    from scripts.add_pdf_embeddings import embed_pdfs
    embed_pdfs("./data/uploads/pdfs")
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

from ..rag_system import RAGSystem
from ..utils.pdf_processor import PDFProcessor
import io


def _existing_pdf_filenames(rag: RAGSystem) -> set[str]:
    """Return set of filenames already stored in vector store as PDFs (either summary or raw)."""
    try:
        collection = rag.chroma_client.get_or_create_collection("ofergpt_memories")
        data = collection.get()
        filenames: set[str] = set()
        if data and data.get("metadatas"):
            for meta in data["metadatas"]:
                if not meta:
                    continue
                t = meta.get("type")
                fn = meta.get("filename")
                if fn and t in ("pdf_document", "pdf_document_raw"):
                    filenames.add(fn)
        return filenames
    except Exception:
        return set()


def embed_pdfs(pdf_dir: str | Path = "./data/uploads/pdfs") -> None:
    """Read PDFs from a directory and embed any that are not yet stored."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        print(f"[PDF] Directory not found: {pdf_dir}")
        return

    pdf_files = [p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"]
    if not pdf_files:
        print("[PDF] No PDF files found – skipping")
        return

    rag = RAGSystem()
    existing = _existing_pdf_filenames(rag)

    processor = PDFProcessor()
    added_files = 0
    total_chunks = 0
    for pdf_path in pdf_files:
        if pdf_path.name in existing:
            continue
        try:
            # Read bytes and process via the same pipeline as app uploads
            with open(pdf_path, "rb") as f:
                data = f.read()
            uploaded_like = io.BytesIO(data)
            # Add getbuffer compatibility if needed (BytesIO supports it)
            pdf_data = processor.process_pdf_file(uploaded_like, pdf_path.name)
            if not pdf_data:
                print(f"[PDF] Skipping {pdf_path.name}: no extractable text")
                continue
            docs = processor.create_pdf_descriptions(pdf_data)
            if not docs:
                print(f"[PDF] No documents created for {pdf_path.name}")
                continue
            # Add to vector store using RAGSystem path
            before = rag.chroma_client.get_or_create_collection("ofergpt_memories").count()
            rag.add_pdf_documents(docs)
            after = rag.chroma_client.get_or_create_collection("ofergpt_memories").count()
            total_chunks += max(0, after - before)
            added_files += 1
        except Exception as e:
            print(f"[PDF] Could not process {pdf_path.name}: {e}")

    if added_files:
        print(f"✅ Added {total_chunks} PDF chunks to vector store ({added_files} new PDF files)")
    else:
        print("[PDF] No new PDFs to embed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed all PDFs in a folder")
    parser.add_argument("--dir", type=str, default="./data/uploads/pdfs", help="Path to directory containing PDFs")
    args = parser.parse_args()
    embed_pdfs(args.dir)


if __name__ == "__main__":
    main()
