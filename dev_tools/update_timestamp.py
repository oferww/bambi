"""Update timestamp metadata for embeddings whose text matches a keyword.

Example usage:

    python scripts/update_timestamp.py --keyword Eiffel --timestamp 2011-10-01T00:00:00
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict

from chromadb import PersistentClient
import piexif
from datetime import datetime

DB_PATH = Path("data/embeddings")
COLLECTION = "ofergpt_memories"


def main():
    parser = argparse.ArgumentParser(description="Patch metadata timestamp for matching documents")
    parser.add_argument("--keyword", required=True, help="Substring to search in document text (case-insensitive)")
    parser.add_argument("--timestamp", required=True, help="ISO timestamp to set")
    args = parser.parse_args()

    client = PersistentClient(str(DB_PATH))
    coll = client.get_or_create_collection(COLLECTION)

    # Get ALL docs (include ids and docs)
    data = coll.get(include=["documents", "metadatas"], limit=None)

    keyword_lower = args.keyword.lower()
    to_update_ids = []
    new_metas: list[Dict[str, Any]] = []

    for idx, doc in enumerate(data["documents"]):
        if keyword_lower in doc.lower():
            meta = data["metadatas"][idx] or {}
            # Update metadata timestamp
            meta["timestamp"] = args.timestamp
            
            # Update EXIF date if we know the file path
            file_path = meta.get("file_path")
            if not file_path and meta.get("filename"):
                # Try to infer path from default photos directory
                candidate = Path("data/photos") / meta["filename"]
                if candidate.exists():
                    file_path = str(candidate)
            if file_path and Path(file_path).exists():
                try:
                    ts_obj = datetime.fromisoformat(args.timestamp)
                    exif_dt = ts_obj.strftime("%Y:%m:%d %H:%M:%S")
                    exif_dict = piexif.load(file_path)
                    if "Exif" not in exif_dict:
                        exif_dict["Exif"] = {}
                    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_dt.encode()
                    piexif.insert(piexif.dump(exif_dict), file_path)
                    print(f"[EXIF] Updated {file_path} -> DateTimeOriginal={exif_dt}")
                except Exception as ex:
                    print(f"[WARN] Could not write EXIF date for {file_path}: {ex}")
            else:
                if not file_path:
                    print("[INFO] No file_path in metadata; skipping EXIF write")
                elif not Path(file_path).exists():
                    print(f"[INFO] file_path not found on disk: {file_path}")

            to_update_ids.append(data["ids"][idx])
            new_metas.append(meta)

    if not to_update_ids:
        print("No documents matched the keyword.")
        return

    coll.update(ids=to_update_ids, metadatas=new_metas)
    print(f"✅ Updated {len(to_update_ids)} documents with timestamp {args.timestamp}")


if __name__ == "__main__":
    main()
