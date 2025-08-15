import os
import json
import re
import csv
from datetime import datetime, timezone
from typing import List, Tuple

# Streamlit's UploadedFile type is duck-typed here (has .name, .type, .getbuffer())
from langchain.schema import Document

from ..rag_system import RAGSystem
from ..utils.pdf_processor import PDFProcessor
from ..utils.photo_processor import PhotoProcessor


### Utilities and type detection ###


def _is_image(name: str, mime: str | None) -> bool:
    name_l = name.lower()
    if mime and mime.startswith("image/"):
        return True
    return name_l.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"))


def _is_pdf(name: str, mime: str | None) -> bool:
    name_l = name.lower()
    return (mime == "application/pdf") or name_l.endswith(".pdf")


def _is_csv(name: str, mime: str | None) -> bool:
    name_l = name.lower()
    return (mime in ("text/csv", "application/vnd.ms-excel")) or name_l.endswith(".csv")


def _to_iso(ts_val) -> str:
    try:
        # Instagram exports are often epoch seconds
        if isinstance(ts_val, (int, float)):
            return datetime.fromtimestamp(int(ts_val), tz=timezone.utc).isoformat()
        if isinstance(ts_val, str) and ts_val.isdigit():
            return datetime.fromtimestamp(int(ts_val), tz=timezone.utc).isoformat()
        # Attempt parsing ISO strings
        return datetime.fromisoformat(str(ts_val)).astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


def _to_iso_date(s: str) -> str:
    """Parse common date formats to YYYY-MM-DD; return empty string on failure."""
    if not s:
        return ""
    try:
        return datetime.fromisoformat(s.strip()).date().isoformat()
    except Exception:
        pass
    try:
        return datetime.strptime(s.strip(), "%m/%d/%Y").date().isoformat()
    except Exception:
        return ""


def _extract_hashtags(text: str) -> list:
    return re.findall(r"#(\w+)", text or "")


### CSV ingestion helpers ###


def _is_imdb_ratings_csv(path: str) -> bool:
    """Heuristically detect if a CSV looks like IMDb ratings export.

    NOTE: Generic CSV ingestion is the default. To re-enable IMDb-specific
    routing, see the commented example in `ingest_csvs_in_uploads()` below.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
        header = [h.strip().lower() for h in first.strip().split(",")]
        required = {"const", "title", "your rating"}
        return required.issubset(set(header))
    except Exception:
        return False


def _ingest_imdb_ratings_csv(path: str, rag: RAGSystem) -> int:
    """Read an IMDb ratings CSV and add one document per row to the vector store."""
    try:
        print(f"[INGEST][IMDB] Reading CSV: {path}", flush=True)
        rows = 0
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                const = (row.get("Const") or row.get("const") or "").strip()
                title = (row.get("Title") or "").strip()
                year = (row.get("Year") or "").strip()
                your_rating = (row.get("Your Rating") or row.get("Your rating") or "").strip()
                imdb_rating = (
                    row.get("IMDb Rating")
                    or row.get("IMDB Rating")
                    or row.get("Imdb Rating")
                    or ""
                ).strip()
                date_rated_raw = (
                    row.get("Date Rated") or row.get("Date rated") or row.get("Date") or ""
                ).strip()
                date_iso = _to_iso_date(date_rated_raw) if date_rated_raw else ""
                genres = (row.get("Genres") or "").strip()
                directors = (row.get("Directors") or "").strip()
                runtime = (row.get("Runtime (mins)") or row.get("Runtime") or "").strip()
                url = (row.get("URL") or row.get("Url") or "").strip()

                if not const and not title:
                    continue

                # Stable id + key
                fname = f"imdb:{const or title}:{date_iso or 'nodate'}"
                base_key = f"imdb:{const}:{date_iso}:{your_rating}"
                import hashlib as _hl
                idem_key = _hl.sha256(base_key.encode("utf-8")).hexdigest()[:16]

                meta = {
                    "type": "imdb",
                    "source": "imdb",
                    "platform": "imdb",
                    "filename": fname,
                    "imdb_id": const,
                    "title": title,
                    "year": year,
                    "your_rating": your_rating,
                    "imdb_rating": imdb_rating,
                    "genres": genres,
                    "directors": directors,
                    "runtime_mins": runtime,
                    "url": url,
                    "date_rated": date_iso or date_rated_raw,
                    "timestamp": date_iso or "",
                    "idempotency_key": idem_key,
                }
                # Store full raw row in metadata for maximum fidelity
                try:
                    meta["imdb_row_json"] = json.dumps(row, ensure_ascii=False)
                except Exception:
                    pass

                # Build a comprehensive content block including all available fields
                header_lines = []
                if title:
                    header_lines.append(f"Title: {title}")
                if year:
                    header_lines.append(f"Year: {year}")
                if const:
                    header_lines.append(f"IMDb ID: {const}")
                if url:
                    header_lines.append(f"URL: {url}")

                # Include all non-empty CSV fields as key: value lines
                details_lines = []
                for k, v in row.items():
                    try:
                        val = v.strip() if isinstance(v, str) else v
                    except Exception:
                        val = v
                    if val is None or (isinstance(val, str) and val == ""):
                        continue
                    details_lines.append(f"{k}: {val}")

                # Final content string (multi-line for readability)
                content = "\n".join(header_lines + details_lines) if (header_lines or details_lines) else (title or const or "IMDb entry")
                docs.append({"content": content, "metadata": meta})
                rows += 1

        if not docs:
            print(f"[INGEST][IMDB] No rows parsed from {path}", flush=True)
            return 0
        print(f"[INGEST][IMDB] Adding {len(docs)} IMDb rows …", flush=True)
        rag.add_document_descriptions(docs)
        print(f"[INGEST][IMDB] Added {len(docs)} rows (post-chunking may differ)", flush=True)
        return rows
    except Exception as e:
        print(f"[INGEST][IMDB] Failed to ingest '{path}': {e}")
        return 0


def _ingest_generic_csv(path: str, rag: RAGSystem) -> int:
    """Ingest any generic CSV by turning each row into a Document.

    - Builds a readable content block from non-empty fields.
    - Adds metadata with type/source/platform='csv', original filename, row index, and headers.
    - Relies on RAGSystem.add_document_descriptions for dedup and splitting.
    Returns number of rows parsed.
    """
    try:
        print(f"[INGEST][CSV] Reading CSV: {path}", flush=True)
        rows_parsed = 0
        docs: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            for i, row in enumerate(reader):
                # Skip completely empty rows
                if not row or not any((v or "").strip() for v in row.values()):
                    continue

                # Try to derive a date/timestamp from common columns
                ts_cols = [
                    "timestamp", "date", "date_only", "date_taken", "date_rated",
                    "created_at", "updated_at", "time", "datetime"
                ]
                ts_val = ""
                for k in ts_cols:
                    v = row.get(k) or row.get(k.title()) or row.get(k.upper())
                    if v:
                        ts_val = _to_iso_date(str(v)) or str(v)
                        break

                # Construct content as multi-line key: value list
                details = []
                for k, v in row.items():
                    try:
                        val = v.strip() if isinstance(v, str) else v
                    except Exception:
                        val = v
                    if val is None or (isinstance(val, str) and val == ""):
                        continue
                    details.append(f"{k}: {val}")
                content = "\n".join(details) if details else f"CSV row {i}"

                meta = {
                    "type": "csv",
                    "source": "csv",
                    "platform": "csv",
                    "filename": f"csv:{os.path.basename(path)}:{i}",
                    "original_csv": os.path.basename(path),
                    "row_index": i,
                    "headers": headers,
                }
                if ts_val:
                    meta["timestamp"] = ts_val

                docs.append({"content": content, "metadata": meta})
                rows_parsed += 1

        if not docs:
            print(f"[INGEST][CSV] No rows parsed from {path}")
            return 0
        print(f"[INGEST][CSV] Adding {len(docs)} rows from {os.path.basename(path)} …", flush=True)
        rag.add_document_descriptions(docs)
        print(f"[INGEST][CSV] Added {len(docs)} rows (post-chunking may differ)")
        return rows_parsed
    except Exception as e:
        print(f"[INGEST][CSV] Failed to ingest '{path}': {e}")
        return 0


### JSON ingestion helpers ###


def _ingest_instagram_posts_json(path: str, rag: RAGSystem) -> int:
    """Ingest Instagram posts JSON, treating each as a 'photo-like' document without an image."""
    try:
        print(f"[INGEST][IG] Reading JSON: {path}", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[INGEST] Failed to read JSON '{path}': {e}")
        return 0

    # Instagram dump variants: list of posts or dict with key
    if isinstance(data, dict):
        posts = data.get("posts") or data.get("ig_posts") or data.get("media") or []
    elif isinstance(data, list):
        posts = data
    else:
        posts = []

    print(f"[INGEST][IG] Parsed posts count: {len(posts)}", flush=True)
    docs = []
    # Use Nominatim (via PhotoProcessor) to resolve human-readable locations
    pp = PhotoProcessor()
    for i, item in enumerate(posts):
        # Heuristic field extraction
        post_id = item.get("id") or item.get("media_id") or item.get("pk") or item.get("uri") or ""
        caption = (
            (item.get("caption") or {}) .get("text") if isinstance(item.get("caption"), dict) else item.get("caption")
        ) or item.get("title") or item.get("description") or ""
        # Only parse coordinates (ignore location names)
        lat = None
        lon = None

        # Preferred: coordinates in media -> media_metadata -> photo_metadata -> exif_data
        media_list = item.get("media") if isinstance(item.get("media"), list) else []
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

        # Fallbacks
        if (lat is None or lon is None) and isinstance(item.get("location"), dict):
            loc = item.get("location") or {}
            lat = loc.get("latitude") or loc.get("lat") or lat
            lon = loc.get("longitude") or loc.get("lng") or lon

        if (lat is None or lon is None) and isinstance(item.get("media_metadata"), dict):
            pm2 = (item.get("media_metadata") or {}).get("photo_metadata") or {}
            exif2 = pm2.get("exif_data")
            if isinstance(exif2, list):
                for d in exif2:
                    if isinstance(d, dict) and ("latitude" in d and "longitude" in d):
                        lat = d.get("latitude", lat)
                        lon = d.get("longitude", lon)
                        break

        # Timestamp only
        ts = item.get("creation_timestamp") or item.get("taken_at") or item.get("timestamp") or item.get("date")
        iso_ts = _to_iso(ts)

        # Build minimal metadata
        filename = f"instagram:{post_id or (iso_ts or '')[:19] or str(i)}"
        meta = {
            "type": "instagram",
            "source": "instagram",
            "platform": "instagram",
            "filename": filename,
            "timestamp": iso_ts,
            "caption": caption,
        }
        # Extract hashtags from caption and attach to metadata/payload
        hashtags = []
        try:
            hashtags = _extract_hashtags(caption)
            if hashtags:
                meta["hashtags"] = hashtags
        except Exception:
            hashtags = []
        if post_id:
            meta["instagram_id"] = post_id
        # attach coordinates if available
        if lat is not None and lon is not None:
            meta["coordinates"] = {"lat": lat, "lon": lon}

        # Resolve human-readable location name using Nominatim
        location_name = None
        if lat is not None and lon is not None:
            try:
                location_name = pp._get_location_name(float(lat), float(lon))
            except Exception:
                location_name = None
        if location_name:
            meta["location_name"] = location_name

        # Idempotency key (prefer stable post_id)
        from ..rag_system import RAGSystem as _RS  # local import to reuse hasher via instance not available
        # lightweight local hash to avoid circular usage of rag method
        import hashlib as _hl, os as _os
        # Build a robust base key to avoid collisions when fields are missing
        base_parts = [
            "instagram",
            (post_id or f"i{i}"),
            (iso_ts or "notime"),
            (caption[:64] or f"nocap{i}"),
            _os.path.basename(path),
        ]
        base_key = ":".join(base_parts)
        idem_key = _hl.sha256(base_key.encode("utf-8")).hexdigest()[:16]
        meta["idempotency_key"] = idem_key
        print(f"[INGEST][IG] Built meta: id={post_id} ts={iso_ts} coords={meta.get('coordinates')} key={idem_key}", flush=True)

        payload = {
            "platform": "instagram",
            "caption": caption,
            "timestamp": iso_ts,
            "coordinates": ({"lat": lat, "lon": lon} if (lat is not None and lon is not None) else None),
        }
        if hashtags:
            payload["hashtags"] = hashtags
        if location_name:
            payload["location_name"] = location_name
        content = json.dumps(payload, ensure_ascii=False)
        docs.append({"content": content, "metadata": meta})

    if not docs:
        print(f"[INGEST][IG] No docs built from {path}", flush=True)
        return 0

    before = len(docs)
    print(f"[INGEST][IG] Adding {before} instagram docs to vector store…", flush=True)
    rag.add_document_descriptions(docs)
    print(f"[INGEST][IG] Added {before} docs (post-chunking may differ)", flush=True)
    return len(docs)


def _ingest_generic_json(path: str, rag: RAGSystem) -> int:
    """Ingest any JSON file.
    - If it looks like an Instagram export, route to _ingest_instagram_posts_json.
    - If top-level is a list: each item becomes a document (stringified JSON).
    - If top-level is a dict: a single document is created.
    Returns number of documents created (or posts for Instagram).
    """
    print(f"[INGEST][JSON] Reading JSON: {path}", flush=True)
    data = None
    try:
        # If it's .jsonl, prefer NDJSON parsing
        if path.lower().endswith('.jsonl'):
            raise ValueError("force_ndjson")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e1:
        # Fallback to NDJSON / per-line JSON objects
        try:
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    # allow trailing commas or stray characters by best-effort strip
                    try:
                        obj = json.loads(s)
                        items.append(obj)
                    except Exception:
                        # Accumulate multi-line objects: simple buffer approach
                        # Read until a parsable JSON object is formed
                        buf = s
                        while True:
                            nxt = f.readline()
                            if not nxt:
                                break
                            buf += "\n" + nxt
                            try:
                                obj = json.loads(buf)
                                items.append(obj)
                                break
                            except Exception:
                                continue
            if items:
                data = items
            else:
                raise
        except Exception as e2:
            print(f"[INGEST][JSON] Failed to parse JSON '{path}': {e1} | NDJSON fallback: {e2}")
            return 0

    # Instagram detection: dict with posts/ig_posts/media OR list of items with instagram-like fields
    try:
        if isinstance(data, dict):
            if any(k in data for k in ("posts", "ig_posts", "media")):
                return _ingest_instagram_posts_json(path, rag)
        if isinstance(data, list) and data:
            sample = data[0]
            if isinstance(sample, dict) and any(k in sample for k in ("caption", "creation_timestamp", "taken_at", "media")):
                return _ingest_instagram_posts_json(path, rag)
    except Exception:
        # fall through to generic
        pass

    docs = []
    base = os.path.basename(path)
    def _extract_ts(obj) -> str:
        ts = None
        if isinstance(obj, dict):
            for k in ("timestamp", "creation_timestamp", "taken_at", "date", "datetime", "time"):
                v = obj.get(k)
                if v:
                    ts = v
                    break
        return _to_iso(ts) if ts is not None else ""

    import hashlib as _hl
    if isinstance(data, list):
        for i, item in enumerate(data):
            try:
                content = json.dumps(item, ensure_ascii=False)
            except Exception:
                content = str(item)
            ts = _extract_ts(item)
            # Build idempotency using filename + index + content hash
            content_hash = _hl.sha256(content[:1024].encode("utf-8")).hexdigest()[:16]
            idem_key = _hl.sha256(f"json:{base}:{i}:{content_hash}:{ts}".encode("utf-8")).hexdigest()[:16]
            meta = {
                "type": "json",
                "source": "json",
                "platform": "json",
                "filename": f"json:{base}:{i}",
                "timestamp": ts,
                "idempotency_key": idem_key,
                "original_json": base,
            }
            docs.append({"content": content, "metadata": meta})
    else:
        # Single document
        try:
            content = json.dumps(data, ensure_ascii=False)
        except Exception:
            content = str(data)
        ts = _extract_ts(data)
        content_hash = _hl.sha256(content[:2048].encode("utf-8")).hexdigest()[:16]
        idem_key = _hl.sha256(f"json:{base}:single:{content_hash}:{ts}".encode("utf-8")).hexdigest()[:16]
        meta = {
            "type": "json",
            "source": "json",
            "platform": "json",
            "filename": f"json:{base}",
            "timestamp": ts,
            "idempotency_key": idem_key,
        }
        docs.append({"content": content, "metadata": meta})

    if not docs:
        print(f"[INGEST][JSON] No docs built from {path}")
        return 0
    print(f"[INGEST][JSON] Adding {len(docs)} JSON docs …", flush=True)
    rag.add_document_descriptions(docs)
    print(f"[INGEST][JSON] Added {len(docs)} docs", flush=True)
    return len(docs)


### Public ingestion entry points ###


def ingest_pdfs_in_uploads(rag: RAGSystem, pdfs_dir: str = "./data/uploads/pdfs") -> int:
    """Process all PDFs in a directory and add to RAG. Returns number of PDFs added."""
    os.makedirs(pdfs_dir, exist_ok=True)
    pdf_processor = PDFProcessor(uploads_dir=pdfs_dir)
    pdfs_added = 0
    try:
        for fname in os.listdir(pdfs_dir):
            if not fname.lower().endswith('.pdf'):
                continue
            fpath = os.path.join(pdfs_dir, fname)
            try:
                with open(fpath, 'rb') as fh:
                    class _F:
                        def __init__(self, name, data):
                            self.name = name
                            self._data = data
                        def getbuffer(self):
                            return self._data
                    buf = fh.read()
                    pseudo = _F(fname, buf)
                    pdf_data = pdf_processor.process_pdf_file(pseudo, fname)
                if pdf_data:
                    docs = pdf_processor.create_pdf_descriptions(pdf_data)
                    if docs:
                        rag.add_pdf_documents(docs)
                        pdfs_added += 1
            except Exception as e:
                print(f"[INGEST] Error processing PDF '{fname}': {e}")
    except Exception as e:
        print(f"[INGEST] Error scanning uploads for PDFs: {e}")
    return pdfs_added


def ingest_csvs_in_uploads(rag: RAGSystem | None = None, csv_dir: str = "./data/uploads/csv") -> int:
    """Embed all CSVs in a directory.
    - Generic ingestion only: every CSV row becomes a document with all fields.
    Returns number of CSV files processed (files with >=1 parsed rows).
    """
    os.makedirs(csv_dir, exist_ok=True)
    count = 0
    if rag is None:
        rag = RAGSystem()
    try:
        for fname in os.listdir(csv_dir):
            if not fname.lower().endswith('.csv'):
                continue
            fpath = os.path.join(csv_dir, fname)
            try:
                rows = _ingest_generic_csv(fpath, rag)
                if rows > 0:
                    count += 1
                # --- OPTIONAL: re-enable themed handlers (IMDb / Locations) ---
                # To route certain CSVs to specialized logic instead of generic ingestion,
                # replace the block above with the following template:
                #
                # if _is_imdb_ratings_csv(fpath):
                #     _ingest_imdb_ratings_csv(fpath, rag)
                #     count += 1
                # elif _looks_like_locations_csv(fpath) and (embed_locations is not None):
                #     embed_locations(fpath)
                #     count += 1
                # else:
                #     rows = _ingest_generic_csv(fpath, rag)
                #     if rows > 0:
                #         count += 1
                # ---------------------------------------------------------------
            except Exception as e:
                print(f"[INGEST] Error embedding CSV '{fname}': {e}")
    except Exception as e:
        print(f"[INGEST] Error scanning uploads for CSVs: {e}")
    return count


def ingest_jsons_in_uploads(rag: RAGSystem, json_dir: str = "./data/uploads/json") -> int:
    """Process any *.json/*.jsonl files in ./data/uploads/json using generic JSON ingestion.
    Returns number of documents ingested.
    """
    os.makedirs(json_dir, exist_ok=True)
    posts_added = 0
    try:
        for fname in os.listdir(json_dir):
            if not fname.lower().endswith(('.json', '.jsonl')):
                continue
            fpath = os.path.join(json_dir, fname)
            try:
                posts_added += _ingest_generic_json(fpath, rag)
            except Exception as e:
                print(f"[INGEST] Error processing JSON '{fname}': {e}")
    except Exception as e:
        print(f"[INGEST] Error scanning uploads for JSON: {e}")
    return posts_added


def ingest_photos_in_uploads(rag: RAGSystem, photos_dir: str = "./data/uploads/photos") -> int:
    """Scan photos directory for images and embed new ones using RAGSystem.auto_sync_from_disk().
    Returns number of newly added photos (best-effort estimate based on filenames before/after).
    """
    try:
        # Count existing photo filenames before
        collection = rag.chroma_client.get_or_create_collection("ofergpt_memories")
        data_before = collection.get()
        names_before = set()
        if data_before and data_before.get("metadatas"):
            for m in data_before["metadatas"]:
                if m and (m.get("type") == "photo" or (m.get("filename",""))):
                    names_before.add(m.get("filename", ""))
        rag.auto_sync_from_disk(photos_dir=photos_dir)
        data_after = collection.get()
        names_after = set()
        if data_after and data_after.get("metadatas"):
            for m in data_after["metadatas"]:
                if m and (m.get("type") == "photo" or (m.get("filename",""))):
                    names_after.add(m.get("filename", ""))
        added = max(0, len([n for n in names_after if n and n not in names_before]))
        return added
    except Exception as e:
        print(f"[INGEST] Error ingesting photos from dir: {e}")
        return 0


### aggregate ingestion ###


def ingest_scan_uploads(rag: RAGSystem) -> Tuple[int, int, int]:
    """
    Scan ./data/uploads subfolders for supported files and ingest them. 
    Useful for uploading files already in the uploads folder.
    - Instagram posts_*.json
    - PDFs (*.pdf)
    - CSVs (*.csv)
    Returns counts tuple.
    """
    photos_added = 0
    pdfs_added = 0
    csvs_processed = 0

    uploads_root = "./data/uploads"
    pdfs_dir = os.path.join(uploads_root, "pdfs")
    csv_dir = os.path.join(uploads_root, "csv")
    json_dir = os.path.join(uploads_root, "json")
    os.makedirs(pdfs_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # PDF processor
    pdf_processor = PDFProcessor(uploads_dir=pdfs_dir)

    try:
        # JSONs (Instagram or generic)
        for fname in os.listdir(json_dir):
            fpath = os.path.join(json_dir, fname)
            if fname.lower().endswith(('.json', '.jsonl')):
                photos_added += _ingest_generic_json(fpath, rag)
        # PDFs
        for fname in os.listdir(pdfs_dir):
            fpath = os.path.join(pdfs_dir, fname)
            low = fname.lower()
            # Instagram JSON
            if low.endswith('.pdf'):
                try:
                    with open(fpath, 'rb') as fh:
                        class _F:  # minimal duck-typed wrapper for PDFProcessor
                            def __init__(self, name, data):
                                self.name = name
                                self._data = data
                            def getbuffer(self):
                                return self._data
                        buf = fh.read()
                        pseudo = _F(fname, buf)
                        pdf_data = pdf_processor.process_pdf_file(pseudo, fname)
                        if pdf_data:
                            docs = pdf_processor.create_pdf_descriptions(pdf_data)
                            if docs:
                                rag.add_pdf_documents(docs)
                                pdfs_added += 1
                except Exception as e:
                    print(f"[INGEST] Error processing PDF '{fname}': {e}")
        # CSVs
        for fname in os.listdir(csv_dir):
            fpath = os.path.join(csv_dir, fname)
            low = fname.lower()
            if low.endswith('.csv'):
                try:
                    if _is_imdb_ratings_csv(fpath):
                        _ingest_imdb_ratings_csv(fpath, rag)
                        csvs_processed += 1
                except Exception as e:
                    print(f"[INGEST] Error embedding CSV '{fname}': {e}")
    except Exception as e:
        print(f"[INGEST] Error scanning uploads dir: {e}")

    return photos_added, pdfs_added, csvs_processed


def ingest_files(uploaded_files: List, rag: RAGSystem) -> Tuple[int, int, int]:
    """
    Lightweight ingestion dispatcher. 
    Useful for uploading files from frontend sidepanel.
    - Saves files into ./data/uploads/{photos,pdfs,csv,json}
    - Images: extract metadata via PhotoProcessor and add document descriptions
    - PDFs: process via PDFProcessor and add summary+raw documents
    - CSVs: call embed_locations (if available)

    Returns: (photos_added, pdfs_added, csvs_processed)
    """
    photos_added = 0
    pdfs_added = 0
    csvs_processed = 0

    os.makedirs("./data/uploads/photos", exist_ok=True)
    os.makedirs("./data/uploads/pdfs", exist_ok=True)
    os.makedirs("./data/uploads/csv", exist_ok=True)
    os.makedirs("./data/uploads/json", exist_ok=True)

    # Prepare processors
    photo_processor = PhotoProcessor(photos_dir="./data/uploads/photos")
    pdf_processor = PDFProcessor(uploads_dir="./data/uploads/pdfs")

    # Group by type
    images = [f for f in uploaded_files if _is_image(getattr(f, "name", ""), getattr(f, "type", None))]
    pdfs = [f for f in uploaded_files if _is_pdf(getattr(f, "name", ""), getattr(f, "type", None))]
    csvs = [f for f in uploaded_files if _is_csv(getattr(f, "name", ""), getattr(f, "type", None))]
    jsons = [f for f in uploaded_files if str(getattr(f, "name", "")).lower().endswith((".json", ".jsonl"))]

    # Process images
    if images:
        new_descs = []
        for img in images:
            fname = img.name
            save_path = os.path.join("./data/uploads/photos", fname)
            # Save file
            with open(save_path, "wb") as out:
                out.write(img.getbuffer())
            # Extract metadata
            meta = photo_processor.extract_metadata(save_path)
            if not meta:
                continue
            # Ensure only metadata is stored; do not persist original file path
            try:
                meta.pop("file_path", None)
            except Exception:
                pass
            meta["type"] = meta.get("type") or "photo"
            content = __import__("json").dumps(meta, ensure_ascii=False)
            new_descs.append({"content": content, "metadata": meta})
            # Delete the original file after processing to avoid storing heavy images on disk
            try:
                os.remove(save_path)
            except Exception as _e:
                print(f"[INGEST] Warning: could not delete image '{save_path}': {_e}")
        if new_descs:
            rag.add_document_descriptions(new_descs)
            photos_added = len(new_descs)

    # Process PDFs
    for pdf in pdfs:
        pdf_data = pdf_processor.process_pdf_file(pdf, pdf.name)
        if not pdf_data:
            continue
        docs = pdf_processor.create_pdf_descriptions(pdf_data)
        if docs:
            rag.add_pdf_documents(docs)
            pdfs_added += 1

    # Process CSVs (locations or IMDb ratings)
    for csv_file in csvs:
        try:
            save_path = os.path.join("./data/uploads/csv", csv_file.name)
            with open(save_path, "wb") as out:
                out.write(csv_file.getbuffer())
            if _is_imdb_ratings_csv(save_path):
                _ingest_imdb_ratings_csv(save_path, rag)
                csvs_processed += 1
        except Exception as e:
            print(f"[INGEST] Error processing CSV '{csv_file.name}': {e}")

    # Process JSONs (Instagram posts exports)
    for js in jsons:
        try:
            save_path = os.path.join("./data/uploads/json", js.name)
            with open(save_path, "wb") as out:
                out.write(js.getbuffer())
            # Route any JSON through the generic handler (auto-detects Instagram)
            photos_added += _ingest_generic_json(save_path, rag)
        except Exception as e:
            print(f"[INGEST] Error processing JSON '{js.name}': {e}")

    # Also scan existing uploads dir for any *.json/.jsonl to catch files added outside UI
    try:
        for fname in os.listdir("./data/uploads/json"):
            if fname.lower().endswith(('.json', '.jsonl')):
                fpath = os.path.join("./data/uploads/json", fname)
                photos_added += _ingest_generic_json(fpath, rag)
    except Exception as e:
        print(f"[INGEST] Error scanning uploads for Instagram JSON: {e}")

    return photos_added, pdfs_added, csvs_processed
