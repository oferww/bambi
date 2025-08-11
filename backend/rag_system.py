import os
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_community.vectorstores.utils import filter_complex_metadata  # Not needed, using custom processing
from typing import List, Dict, Any
from .utils.photo_processor import PhotoProcessor
import json
import hashlib
import csv
import cohere
import re
 
 

# Comprehensive ChromaDB telemetry disabling
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_AUTH_CREDENTIALS_FILE"] = ""
os.environ["CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER"] = ""
os.environ["CHROMA_SERVER_NOFILE"] = ""

# Additional environment variables to disable telemetry
import sys

# Monkey patch before any ChromaDB imports
def disable_telemetry():
    """Completely disable ChromaDB telemetry by patching at import time"""
    try:
        # Mock posthog entirely before it gets imported
        class MockPosthog:
            def __init__(self, *args, **kwargs):
                pass
            def capture(self, *args, **kwargs):
                pass
            def identify(self, *args, **kwargs):
                pass
            def reset(self, *args, **kwargs):
                pass
            def __call__(self, *args, **kwargs):
                return self
        
        # Mock the entire posthog module
        import types
        mock_posthog_module = types.ModuleType('posthog')
        mock_posthog_module.Posthog = MockPosthog
        mock_posthog_module.capture = lambda *args, **kwargs: None
        mock_posthog_module.identify = lambda *args, **kwargs: None
        mock_posthog_module.reset = lambda *args, **kwargs: None
        
        # Insert into sys.modules to prevent real posthog import
        sys.modules['posthog'] = mock_posthog_module
        
        print("🔇 ChromaDB telemetry completely disabled via posthog mock", flush=True)
        return True
        
    except Exception as e:
        print(f"⚠️  Could not disable telemetry: {e}", flush=True)
        return False

# Apply telemetry disabling
disable_telemetry()

class RAGSystem:
    """RAG system using Cohere embeddings and ChromaDB for storage."""
    
    def __init__(self, embeddings_dir: str = "./data/embeddings"):
        self.embeddings_dir = embeddings_dir
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Initialize Cohere embeddings
        self.embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=embeddings_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            client=self.chroma_client,
            embedding_function=self.embeddings,
            collection_name="ofergpt_memories"
        )
        
        # Text splitter for chunking (env-configurable)
        chunk_size = int(os.getenv("OFERGPT_RAG_CHUNK_SIZE", "1500"))
        chunk_overlap = int(os.getenv("OFERGPT_RAG_CHUNK_OVERLAP", "250"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # Optional: auto-sync photos from disk on initialization (disabled by default)
        try:
            import os as _os
            if _os.getenv("OFERGPT_RAG_AUTOSYNC", "0") == "1":
                self.auto_sync_from_disk()
        except Exception as _e:
            print(f"[INIT] Skipping auto_sync_from_disk due to error: {_e}")

    def _collect_existing_idempotency_keys(self) -> set:
        """Collect existing idempotency keys from the collection for dedupe."""
        keys = set()
        try:
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            data = collection.get()
            if data and data.get("metadatas"):
                for meta in data["metadatas"]:
                    if not meta:
                        continue
                    k = meta.get("idempotency_key")
                    if k:
                        keys.add(k)
        except Exception as e:
            print(f"[DEDUPE] Could not list existing keys: {e}")
        return keys

    def _hash_str(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
    
    def _process_metadata_for_chromadb(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata to be ChromaDB-compatible while preserving location info."""
        processed_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, dict):
                # Handle location dictionary specifically
                if key == "location" and isinstance(value, dict):
                    processed_metadata["latitude"] = str(value.get("latitude", ""))
                    processed_metadata["longitude"] = str(value.get("longitude", ""))
                    processed_metadata["coordinates"] = str(value.get("coordinates", ""))
                    processed_metadata["location_name"] = str(value.get("location_name", ""))
                # Handle coordinates dict placed at top-level metadata (preferred for Instagram)
                elif key == "coordinates" and isinstance(value, dict):
                    lat_val = value.get("lat")
                    lon_val = value.get("lon")
                    # Store helper flat fields for easy querying/reading
                    if lat_val is not None:
                        processed_metadata["lat"] = str(lat_val)
                    if lon_val is not None:
                        processed_metadata["lon"] = str(lon_val)
                    # Store a normalized JSON string for coordinates
                    try:
                        processed_metadata["coordinates"] = json.dumps({"lat": lat_val, "lon": lon_val}, ensure_ascii=False)
                    except Exception:
                        processed_metadata["coordinates"] = str(value)
                # Handle camera_info dictionary
                elif key == "camera_info" and isinstance(value, dict):
                    for camera_key, camera_value in value.items():
                        processed_metadata[f"camera_{camera_key}"] = str(camera_value)
                else:
                    # Convert other dictionaries to strings
                    processed_metadata[key] = str(value)
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to strings
                processed_metadata[key] = str(value)
            elif isinstance(value, (str, int, float, bool)):
                # Keep simple types as-is
                processed_metadata[key] = value
            else:
                # Convert any other complex types to strings
                processed_metadata[key] = str(value)
        
        # Add a normalized date_only (YYYY-MM-DD) if available and not already present
        try:
            if "date_only" not in processed_metadata:
                # Prefer explicit date_only, then other common date fields
                candidates = [
                    metadata.get("date_only"),
                    metadata.get("date_rated"),
                    metadata.get("date_taken"),
                    metadata.get("timestamp"),
                    metadata.get("visit_date"),
                    metadata.get("date"),
                ]
                date_only = ""
                for v in candidates:
                    if not v:
                        continue
                    s = str(v).strip()
                    # If looks like ISO with date at the front: YYYY-MM-DD*
                    if len(s) >= 10 and s[4] == '-' and s[7] == '-':
                        date_only = s[:10]
                        break
                if date_only:
                    processed_metadata["date_only"] = date_only
        except Exception:
            # Best-effort; ignore errors
            pass
        
        return processed_metadata
    
    def _get_existing_filenames(self) -> set:
        """Get set of filenames already present in the vector store (for any document type)."""
        try:
            # Get all documents directly from ChromaDB collection
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            result = collection.get()
            
            existing_files = set()
            if result and 'metadatas' in result:
                for metadata in result['metadatas']:
                    if metadata and 'filename' in metadata:
                        existing_files.add(metadata['filename'])
            
            return existing_files
        except Exception as e:
            print(f"Error getting existing files: {e}")
            return set()
    
    def _dump_embeddings_csv(self, csv_path: str | None = None) -> None:
        """Write current collection embeddings summary to a CSV under data/ for easy inspection."""
        # Resolve default path lazily to project root, but write under data/
        if csv_path is None:
            project_root = os.getenv("PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            csv_path = os.path.join(project_root, "data", "embeddings_dump.csv")
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            data = collection.get()
            if not data:
                return
            with open(csv_path, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Keep legacy flat columns and add a flexible metadata_json column for richer data
                writer.writerow(["id", "source", "location", "timestamp", "text", "metadata_json"])
                rows = 0
                for doc_id, content, metadata in zip(data["ids"], data["documents"], data["metadatas"]):
                    location = metadata.get("location_name") or metadata.get("coordinates") or ""
                    ts = metadata.get("timestamp") or metadata.get("date_taken") or ""
                    source = metadata.get("type", "")
                    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
                    writer.writerow([doc_id, source, location, ts, content.replace("\n", " ")[:1000], metadata_json])
                    rows += 1
            print(f"[CSV] Embeddings dump updated ({rows} rows) -> {csv_path}", flush=True)
        except Exception as e:
            print(f"⚠️  Could not dump embeddings CSV: {e}")

    def add_document_descriptions(self, doc_descriptions: List[Dict[str, Any]]):
        """Add generic document descriptions (photos, Instagram, IMDb, CSV rows, PDFs, etc.) to the vector store with deduplication."""
        documents = []

        # Existing dedupe by filename and idempotency key
        existing_files = self._get_existing_filenames()
        existing_keys = self._collect_existing_idempotency_keys()
        
        for desc in doc_descriptions:
            metadata_in = desc["metadata"]
            filename = metadata_in.get("filename", "")
            doc_type = str(metadata_in.get("type") or metadata_in.get("platform") or metadata_in.get("source") or "document").lower()
            
            # Skip if this document already exists by filename (only when filename exists)
            if filename and filename in existing_files:
                print(f"Skipping duplicate by filename: {filename}")
                continue
            # Compute idempotency key (prefer provided; otherwise build from best available data)
            size = str(metadata_in.get("file_size", ""))
            dt = metadata_in.get("date_taken") or metadata_in.get("timestamp") or metadata_in.get("date") or metadata_in.get("date_rated") or ""
            idem_key = metadata_in.get("idempotency_key")
            if not idem_key:
                if filename:
                    base_key = f"{doc_type}:{filename}:{size}:{dt}"
                else:
                    # Generic fallback: hash the content with type and timestamp
                    content_sample = (desc.get("content") or "")[:512]
                    content_hash = self._hash_str(content_sample)
                    base_key = f"{doc_type}:{content_hash}:{dt}"
                idem_key = self._hash_str(base_key)
            if idem_key in existing_keys:
                label = filename if filename else idem_key
                print(f"Skipping by idempotency_key for doc: {label}")
                continue

            # Process metadata to keep location info in ChromaDB-compatible format
            metadata_in["idempotency_key"] = idem_key
            processed_metadata = self._process_metadata_for_chromadb(metadata_in)
            
            # Create document with metadata
            doc = Document(
                page_content=desc["content"],
                metadata=processed_metadata
            )
            documents.append(doc)
        
        if not documents:
            print("No new documents to add - refreshing CSV anyway")
            self._dump_embeddings_csv()
            return
        
        # Split documents into chunks (but keep Instagram and IMDb docs unsplit to preserve per-item alignment)
        split_docs = []
        try:
            for d in documents:
                plat = str(d.metadata.get("platform") or d.metadata.get("source") or d.metadata.get("type") or "").lower()
                if plat in ("instagram", "imdb"):
                    # Keep a single document per Instagram post or IMDb row
                    split_docs.append(d)
                else:
                    split_docs.extend(self.text_splitter.split_documents([d]))
        except Exception:
            # Fallback to original behavior if anything goes wrong
            split_docs = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vectorstore.add_documents(split_docs)
        
        print(f"Added {len(split_docs)} document chunks to vector store ({len(documents)} new docs)", flush=True)
        # update CSV dump
        self._dump_embeddings_csv()
    
    def add_text_memories(self, memories: List[str]):
        """Add text memories to the vector store."""
        documents = []

        existing_keys = self._collect_existing_idempotency_keys()

        for i, memory in enumerate(memories):
            raw_key = f"text_memory:{i}:{self._hash_str(memory[:2048])}"
            if raw_key in existing_keys:
                print(f"Skipping duplicate text memory {i} by idempotency_key")
                continue
            metadata = {"type": "text_memory", "id": i, "idempotency_key": raw_key}
            filtered_metadata = self._process_metadata_for_chromadb(metadata)

            doc = Document(
                page_content=memory,
                metadata=filtered_metadata
            )
            documents.append(doc)
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vectorstore.add_documents(split_docs)
        
        print(f"Added {len(split_docs)} text memory chunks to vector store")
        self._dump_embeddings_csv()
    
    def auto_sync_from_disk(self, photos_dir: str = "./data/uploads/photos") -> None:
        """Embed new photos in the data folder and enrich missing metadata for existing ones."""
        try:
            photo_processor = PhotoProcessor(photos_dir)
            existing_files = self._get_existing_filenames()
            new_descs = []

            # Pass 1: embed new photos
            for fname in os.listdir(photos_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")):
                    continue
                if fname in existing_files:
                    continue
                file_path = os.path.join(photos_dir, fname)
                meta = photo_processor.extract_metadata(file_path)
                if not meta:
                    # rejected by processor (e.g., no location)
                    continue
                # Ensure only metadata is stored; do not persist original file path
                try:
                    meta.pop("file_path", None)
                except Exception:
                    pass
                meta["type"] = meta.get("type") or "photo"
                content = json.dumps(meta, ensure_ascii=False)
                new_descs.append({"content": content, "metadata": meta})
                # Delete the original file after processing to avoid storing heavy images on disk
                try:
                    os.remove(file_path)
                except Exception as _e:
                    print(f"[SYNC] Warning: could not delete image '{file_path}': {_e}")

            if new_descs:
                print(f"[SYNC] Adding {len(new_descs)} new photos to vector store", flush=True)
                self.add_document_descriptions(new_descs)
            else:
                print("[SYNC] No new photos to embed", flush=True)
        except Exception as e:
            print(f"[SYNC] Error in auto_sync_from_disk: {e}", flush=True)

    def add_pdf_documents(self, pdf_descriptions: List[Dict[str, Any]]):
        """Add PDF documents to the vector store."""
        documents = []

        existing_keys = self._collect_existing_idempotency_keys()

        for desc in pdf_descriptions:
            meta_in = desc["metadata"]
            filename = meta_in.get("filename", "")
            variant = meta_in.get("variant") or meta_in.get("subtype") or "raw"
            # Optional page or slice hints
            page = str(meta_in.get("page", ""))
            base_key = f"pdf:{filename}:{variant}:{page}"
            idem_key = meta_in.get("idempotency_key") or self._hash_str(base_key)
            if idem_key in existing_keys:
                print(f"Skipping duplicate PDF part by idempotency_key: {filename} ({variant})")
                continue
            meta_in["idempotency_key"] = idem_key

            # Process metadata to keep it ChromaDB-compatible
            processed_metadata = self._process_metadata_for_chromadb(meta_in)

            # Create document with metadata
            doc = Document(
                page_content=desc["content"],
                metadata=processed_metadata
            )
            documents.append(doc)
        
        if not documents:
            print("No PDF documents to add – refreshing CSV anyway")
            self._dump_embeddings_csv()
            return
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vectorstore.add_documents(split_docs)
        
        print(f"Added {len(split_docs)} PDF document chunks to vector store")
        self._dump_embeddings_csv()
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content with typo/misspelling robustness.

        Uses Cohere to spell-correct the query for retrieval, then searches using
        the corrected query first (preferred) and the original query second. Merges
        and de-duplicates results, preferring corrected-query hits. No reranking.
        """
        try:
            # 1) Spell-correct query via Cohere (best-effort)
            corrected = query
            try:
                api_key = os.getenv("COHERE_API_KEY")
                if api_key:
                    c = cohere.Client(api_key=api_key)
                    system_instr = (
                        "Correct spelling and typos in the user query for information retrieval. "
                        "Return only the corrected query as plain text with no quotes or extra words."
                        "Do not correct the word 'Ofer' or anything similar, it is the name of the person"
                    )
                    model_name = os.getenv("COHERE_CHAT_MODEL", "command-a-vision-07-2025")
                    resp = c.chat(model=model_name, message=f"{system_instr}\n\nQuery: {query}", temperature=0)
                    if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
                        raw = resp.text.strip()
                        # Helper to sanitize and extract a single corrected query
                        def _extract_single_query(text: str, original: str) -> str:
                            t = text.strip()                            # Remove common prefixes/labels
                            for prefix in [
                                "Corrected query:",
                                "Correction:",
                                "Corrected:",
                                "Query:",
                            ]:
                                if t.lower().startswith(prefix.lower()):
                                    t = t[len(prefix):].strip()
                            # Strip wrapping quotes/backticks/brackets
                            t = t.strip().strip("`")
                            if t.startswith("[") and t.endswith("]"):
                                inner = t[1:-1]
                                parts = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
                                candidates = parts if parts else [t]
                            else:
                                lines = [ln.strip().strip("-• ").strip("'\"") for ln in t.splitlines() if ln.strip()]
                                candidates = lines if len(lines) > 1 else [t]

                            # Choose candidate with highest token overlap to original; tie-break: shorter text
                            orig_tokens = set(original.lower().split())
                            best = None
                            best_score = -1
                            for cand in candidates:
                                tokens = set(cand.lower().split())
                                score = len(tokens & orig_tokens)
                                if best is None or score > best_score or (score == best_score and len(cand) < len(best)):
                                    best = cand
                                    best_score = score
                            return (best or original).strip("'\"")

                        corrected = _extract_single_query(raw, query)
            except Exception as ce:
                print(f"[RAG] Spell-correction skipped due to error: {ce}")

            # Normalize protected tokens: ensure 'Ofer' is preserved exactly
            try:
                if corrected:
                    corrected = re.sub(r"\bofer\b", "Ofer", corrected, flags=re.IGNORECASE)
            except Exception:
                pass

            # 2) Build query list: corrected first (preferred), then original if different
            queries = [corrected] if corrected == query else [corrected, query]
            print(f"[RAG] Using queries: {queries}")
            # 3) Perform searches and merge results with preference for corrected pass
            merged: Dict[str, Dict[str, Any]] = {}

            def _doc_key(doc: Document) -> str:
                meta = doc.metadata or {}
                key = (
                    meta.get("idempotency_key")
                    or meta.get("filename")
                    or hashlib.md5(
                        (doc.page_content[:200] + json.dumps(meta, sort_keys=True, ensure_ascii=False)).encode("utf-8")
                    ).hexdigest()
                )
                return key

            for idx, q in enumerate(queries):
                try:
                    results = self.vectorstore.similarity_search_with_score(q, k=k)
                except Exception as inner_e:
                    print(f"[RAG] Search failed for query variant {idx}: {inner_e}")
                    continue

                for doc, score in results:
                    key = _doc_key(doc)
                    entry = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": float(score),
                        "_preferred": (idx == 0),  # corrected pass first
                    }
                    if key in merged:
                        # Keep the better score (lower is better for distances) and keep preferred if any
                        existing = merged[key]
                        if entry["similarity_score"] < existing["similarity_score"]:
                            existing.update(entry)
                        else:
                            # If scores are equal or worse, but current is preferred and existing isn't, mark preferred
                            if entry["_preferred"] and not existing.get("_preferred"):
                                existing["_preferred"] = True
                    else:
                        merged[key] = entry

            # 4) Sort: prefer corrected-pass results, then by ascending distance/score
            merged_list = list(merged.values())
            merged_list.sort(key=lambda x: (not x.get("_preferred", False), x.get("similarity_score", 1e9)))

            # 5) Trim to top-k and drop helper field
            final = []
            for item in merged_list[:k]:
                item.pop("_preferred", None)
                final.append(item)

            return final
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
 
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection."""
        try:
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            count = collection.count()
            result = collection.get()
            metadatas = result.get('metadatas', []) if result else []

            # Count unique photos, pdfs, and memories
            photo_filenames = set()
            pdf_filenames = set()
            memory_ids = set()
            image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
            num_locations = 0
            for meta in metadatas:
                if not meta:
                    continue
                t = meta.get('type', '')
                
                # Identify photos by type or filename extension
                if t == 'photo' or (meta.get('filename', '') and meta['filename'].lower().endswith(image_exts)):
                    if meta.get('filename'):  # Only count if we have a filename
                        photo_filenames.add(meta['filename'])
                elif t == 'pdf_document' and meta.get('filename'):
                    pdf_filenames.add(meta['filename'])
                elif t == 'text_memory':
                    memory_ids.add(meta.get('id', ''))
                elif t == 'location':
                    num_locations += 1

            num_photos = len(photo_filenames)
            num_pdfs = len(pdf_filenames)
            num_memories = len(memory_ids)
            total_docs = num_photos + num_pdfs + num_memories + num_locations

            return {
                "total_rag_chunks": count,
                "num_photos": num_photos,
                "num_pdfs": num_pdfs,
                "num_memories": num_memories,
                "num_locations": num_locations,
                "total_docs": total_docs
            }
        except Exception as e:
            return {"error": str(e)}
    

    def clean_non_english_locations(self):
        """Remove documents with non-English location names and re-process."""
        try:
            # Get all documents
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            result = collection.get()
            
            if not result or not result.get('metadatas'):
                print("No documents found in collection")
                return
            
            # Find documents with non-English location names
            non_english_docs = []
            doc_data = list(zip(result['ids'], result['documents'], result['metadatas']))
            
            for doc_id, content, metadata in doc_data:
                if metadata and 'location_name' in metadata:
                    location_name = metadata['location_name']
                    # Check if location name contains non-ASCII characters (Thai, etc.)
                    try:
                        location_name.encode('ascii')
                    except UnicodeEncodeError:
                        non_english_docs.append(doc_id)
                        print(f"Found non-English location: {location_name}")
            
            # Remove non-English documents
            if non_english_docs:
                collection.delete(ids=non_english_docs)
                print(f"Removed {len(non_english_docs)} documents with non-English locations")
                print("Please re-process your photos to get English location names")
            else:
                print("No non-English location names found")
                
        except Exception as e:
            print(f"Error cleaning non-English locations: {e}")
    
    def extract_all_locations(self) -> List[Dict[str, Any]]:
        """Extract all unique locations from photos in the vector store."""
        try:
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            result = collection.get()
            
            if not result or not result.get('metadatas'):
                return []
            
            locations = []
            seen_locations = set()
            
            for metadata in result['metadatas']:
                if not metadata:
                    continue
                
                # Check for location data in metadata
                location_name = metadata.get('location_name')
                if location_name and location_name.lower() not in seen_locations:
                    seen_locations.add(location_name.lower())
                    
                    location_data = {
                        'location_name': location_name,
                        'latitude': metadata.get('latitude', ''),
                        'longitude': metadata.get('longitude', ''),
                        'coordinates': metadata.get('coordinates', ''),
                        'source': 'existing_photos',
                        'confidence': 'high'
                    }
                    
                    # Try to get date from metadata
                    date_taken = metadata.get('date_taken', '')
                    filename = metadata.get('filename', '')
                    
                    location_data['visit_date'] = date_taken
                    location_data['photo_filename'] = filename
                    
                    locations.append(location_data)
            
            print(f"Extracted {len(locations)} unique locations from vector store")
            return locations
            
        except Exception as e:
            print(f"Error extracting locations: {e}")
            return []

    def update_photo_location(self, filename: str, new_location_name: str, new_coordinates: str = ""):
        """Update the location for a specific photo in the vector store."""
        try:
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            result = collection.get()
            
            updated_count = 0
            for doc_id, metadata in zip(result['ids'], result['metadatas']):
                if metadata and metadata.get('filename') == filename:
                    # Update the metadata
                    metadata['location_name'] = new_location_name
                    if new_coordinates:
                        metadata['coordinates'] = new_coordinates
                        # Parse coordinates if provided
                        if ', ' in new_coordinates:
                            lat, lon = new_coordinates.split(', ')
                            metadata['latitude'] = lat.strip()
                            metadata['longitude'] = lon.strip()
                    
                    # Update in ChromaDB
                    collection.update(
                        ids=[doc_id],
                        metadatas=[metadata]
                    )
                    updated_count += 1
                    print(f"Updated location for {filename}: {new_location_name}")
            
            if updated_count == 0:
                print(f"No photo found with filename: {filename}")
            else:
                print(f"Updated {updated_count} documents for {filename}")
                
        except Exception as e:
            print(f"Error updating photo location: {e}")

    def clear_vector_store(self):
        """Clear all data from the vector store and re-initialize an empty collection so subsequent operations don't fail."""
        try:
            # Delete if it exists (Chroma ignores if missing)
            try:
                self.chroma_client.delete_collection("ofergpt_memories")
                print("🗑️  Vector store collection deleted", flush=True)
            except Exception as inner:
                print(f"[WARN] delete_collection: {inner}", flush=True)
            
            # Re-create empty collection and refresh vectorstore handle
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name="ofergpt_memories",
                embedding_function=self.embeddings,
            )
            print("✅ Empty collection re-created", flush=True)
        except Exception as e:
            print(f"Error clearing vector store: {e}", flush=True)

    def delete_by_filter(self, where: dict) -> int:
        """Delete documents from the collection matching a metadata filter. Returns number deleted (best-effort)."""
        try:
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            # Pre-count matching ids
            to_delete = collection.get(where=where, include=["metadatas", "ids"]) or {}
            ids = to_delete.get("ids", []) or []
            count = len(ids)
            collection.delete(where=where)
            print(f"[DELETE] Deleted {count} docs where={where}", flush=True)
            return count
        except Exception as e:
            print(f"[DELETE] Error deleting where={where}: {e}", flush=True)
            return 0

    def fix_existing_photo_locations(self):
        """Fix existing photos that have coordinates instead of location names."""
        try:
            from .utils.photo_processor import PhotoProcessor
            
            print("[FIX] Starting to fix existing photo locations...", flush=True)
            
            # Get all documents
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            result = collection.get()
            
            if not result or not result.get('metadatas'):
                print("[FIX] No documents found in collection", flush=True)
                return
            
            photo_processor = PhotoProcessor()
            fixed_count = 0
            total_photos = 0
            
            # Process each document
            for doc_id, content, metadata in zip(result['ids'], result['documents'], result['metadatas']):
                if not metadata:
                    continue
                
                # Only process photos
                if metadata.get('type') != 'photo':
                    continue
                
                total_photos += 1
                filename = metadata.get('filename', '')
                location_name = metadata.get('location_name', '')
                
                # Check if location is coordinates (contains numbers and commas)
                if location_name and ',' in location_name and any(c.isdigit() for c in location_name):
                    print(f"[FIX] Found photo with coordinates: {filename} -> {location_name}", flush=True)
                    
                    try:
                        # Parse coordinates
                        coords = location_name.split(',')
                        if len(coords) == 2:
                            lat = float(coords[0].strip())
                            lon = float(coords[1].strip())
                            
                            print(f"[FIX] Parsed coordinates: lat={lat}, lon={lon}", flush=True)
                            
                            # Try to get proper location name
                            new_location_name = photo_processor._get_location_name(lat, lon)
                            
                            if new_location_name and new_location_name != location_name:
                                # Update the metadata
                                metadata['location_name'] = new_location_name
                                
                                # Update in ChromaDB
                                collection.update(
                                    ids=[doc_id],
                                    metadatas=[metadata]
                                )
                                
                                print(f"[FIX] ✅ Fixed {filename}: {location_name} -> {new_location_name}", flush=True)
                                fixed_count += 1
                            else:
                                print(f"[FIX] ⚠️ Could not improve location for {filename}: {location_name}", flush=True)
                        else:
                            print(f"[FIX] ⚠️ Invalid coordinate format for {filename}: {location_name}", flush=True)
                            
                    except Exception as e:
                        print(f"[FIX] ❌ Error fixing {filename}: {e}", flush=True)
                else:
                    print(f"[FIX] Skipping {filename} - already has location name: {location_name}", flush=True)
            
            print(f"[FIX] Completed! Fixed {fixed_count} out of {total_photos} photos", flush=True)
            
        except Exception as e:
            print(f"[FIX] Error in fix_existing_photo_locations: {e}", flush=True)
