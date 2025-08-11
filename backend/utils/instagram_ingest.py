"""Instagram ingest utilities for Bambi.

This module parses the JSON archive you download from Instagram (“Download Your Information” → JSON) and ingests the post metadata into the existing `RAGSystem`.

Typical usage (inside the container):

```python
from utils.instagram_ingest import ingest_instagram_archive
from rag_system import RAGSystem

rag = RAGSystem()
# Perform normal RAG initialisation here …

ingest_instagram_archive("/data/instagram_archive.zip", rag)
```

Alternatively run as a script:

```bash
python -m utils.instagram_ingest /path/to/archive.zip
```
"""

from __future__ import annotations

import json
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Try to reuse helpers if available; provide fallback for cleaner only
try:
    from .photo_processor import clean_location_name  # type: ignore
except Exception:
    def clean_location_name(s: str) -> str:  # type: ignore
        """Minimal fallback: trim whitespace and collapse spaces."""
        try:
            txt = str(s).strip()
            # Remove excessive whitespace
            txt = " ".join(txt.split())
            return txt
        except Exception:
            return str(s)
from .photo_processor import PhotoProcessor  # type: ignore
from ..rag_system import RAGSystem

# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def ingest_instagram_archive(archive_path: str | Path, rag: RAGSystem) -> int:
    """Extract a JSON Instagram archive ZIP and ingest posts into the vector store.

    Parameters
    ----------
    archive_path : str | Path
        Path to the ZIP file downloaded from Instagram (JSON format).
    rag : RAGSystem
        Instance of the already-initialised RAGSystem.

    Returns
    -------
    int
        Number of posts added.
    """
    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    # Create a temp dir to avoid polluting working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmpdir)

        root = Path(tmpdir)

        # Instagram splits posts into posts_1.json, posts_2.json …
        post_files = sorted(root.glob("posts*.json"))
        if not post_files:
            raise RuntimeError("No posts_*.json found inside archive – did you choose JSON when exporting?")

        docs: List[Dict[str, str]] = []
        for pf in post_files:
            with pf.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for post in data.get("posts", []):
                ts = datetime.fromtimestamp(post.get("taken_at", 0))
                ts_readable = ts.strftime("%B %d, %Y") if ts.year > 1970 else "Unknown date"
                caption_text = (
                    post.get("caption", {}) or {}
                ).get("text", "").strip()

                # Attempt to extract coordinates from media EXIF
                lat = None
                lon = None
                media_list = post.get("media") if isinstance(post.get("media"), list) else []
                for m in media_list:
                    if not isinstance(m, dict):
                        continue
                    mm = m.get("media_metadata") or {}
                    pm = mm.get("photo_metadata") or {}
                    exif_list = pm.get("exif_data")
                    if isinstance(exif_list, list) and exif_list:
                        for d in exif_list:
                            if isinstance(d, dict) and ("latitude" in d and "longitude" in d):
                                lat = d.get("latitude", lat)
                                lon = d.get("longitude", lon)
                                break
                    if lat is not None and lon is not None:
                        break

                # Resolve a human-readable location from coordinates; fall back to title if present
                loc_title = (post.get("location") or {}).get("title", "")
                loc_clean = clean_location_name(loc_title) if loc_title else ""
                location_name = loc_clean
                try:
                    if lat is not None and lon is not None:
                        pp = PhotoProcessor()
                        resolved = pp._get_location_name(float(lat), float(lon))
                        if resolved:
                            location_name = resolved
                except Exception:
                    pass

                content_lines = [f"Instagram post on {ts_readable}."]
                if location_name:
                    content_lines.append(f"Location: {location_name}.")
                if caption_text:
                    content_lines.append(f"Caption: {caption_text}")
                content = " ".join(content_lines)

                meta = {
                    "source": "instagram",
                    "timestamp": ts_readable,
                    "caption": caption_text,
                    "instagram_id": post.get("id"),
                }
                if location_name:
                    meta["location_name"] = location_name
                if lat is not None and lon is not None:
                    meta["coordinates"] = {"lat": lat, "lon": lon}

                docs.append({"content": content, "metadata": meta})

        if not docs:
            return 0

        # Add to vector store
        rag.add_documents_bulk(docs)
        return len(docs)

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from ..rag_system import RAGSystem

    parser = argparse.ArgumentParser(description="Ingest Instagram archive into Bambi knowledge base.")
    parser.add_argument("archive", help="Path to Instagram archive ZIP (JSON format)")
    args = parser.parse_args()

    rs = RAGSystem()
    count = ingest_instagram_archive(args.archive, rs)
    print(f"Added {count} Instagram posts to vector store.")
