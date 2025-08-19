import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_community.vectorstores.utils import filter_complex_metadata  # Not needed, using custom processing
from typing import List, Dict, Any
from .utils.photo_processor import PhotoProcessor
import json
import hashlib
import csv
import re
import difflib
from .utils.key_bank import get_keybank
 
 

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
        # Optional API logger injected by chatbot: callable(api_type, which_key, note)
        self.api_logger = None
        os.makedirs(embeddings_dir, exist_ok=True)
        # Initialize KeyBank for rotating Cohere chat keys
        self._keybank = get_keybank()
        
        # Initialize Cohere embeddings (use only EMBED or CHAT keys, no generic COHERE_API_KEY)
        api_key_embed = os.getenv("COHERE_API_KEY_EMBED")
        api_key_chat = os.getenv("COHERE_API_KEY_CHAT")
        self.cohere_key_embed = api_key_embed 
        self.cohere_key_chat = api_key_chat
        if not self.cohere_key_embed or not self.cohere_key_chat:
            raise ValueError(
                "Missing Cohere API key. Set COHERE_API_KEY_EMBED and COHERE_API_KEY_CHAT."
            )
        self.embedding_model_name = "embed-english-v3.0"
        self.embeddings = CohereEmbeddings(
            model=self.embedding_model_name,
            cohere_api_key=self.cohere_key_embed
        )
        # Separate embedding client for QUERY-TIME embeddings using the chat key,
        # while keeping the same embedding model to preserve vector space compatibility
        self.query_embeddings = CohereEmbeddings(
            model=self.embedding_model_name,
            cohere_api_key=self.cohere_key_chat
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
        # Parallel vector store handle for queries that uses the query-time embedding function
        self.vectorstore_query = Chroma(
            client=self.chroma_client,
            embedding_function=self.query_embeddings,
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
                self.auto_sync_photos_from_disk()
        except Exception as _e:
            print(f"[INIT] Skipping auto_sync_photos_from_disk due to error: {_e}")


### Helper utilities ###


    def _hash_str(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

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

    @staticmethod
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

    @staticmethod
    def _extract_single_query(text: str, original: str) -> str:
        # Helper to sanitize and extract a single corrected query
        t = text.strip()
        # Remove common prefixes/labels
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

        # Choose candidate with highest token overlap to original
        orig_tokens = set(original.lower().split())
        best = None
        best_score = -1
        for cand in candidates:
            tokens = set(cand.lower().split())
            score = len(tokens & orig_tokens)
            if best is None or score > best_score:
                best = cand
                best_score = score
        return (best or original).strip("'\"")

    def _is_proper_like(tok: str) -> bool:
        # Guard against introducing new proper-name tokens not present in original
        if not tok or not any(c.isalpha() for c in tok):
            return False
        # Titlecase (John), ALLCAPS (IBM), or Mixed with leading capital
        return tok[:1].isupper() or (tok.isupper() and len(tok) > 1)
            

    ### Document/Photo/PDF processing ###


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

    def auto_sync_photos_from_disk(self, photos_dir: str = "./data/uploads/photos") -> None:
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
            print(f"[SYNC] Error in auto_sync_photos_from_disk: {e}", flush=True)


    ### Search Query ###
    

    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content with typo/misspelling robustness.

        Uses Cohere to spell-correct the query for retrieval, then searches using
        the corrected query first (preferred) and the original query second. Merges
        and de-duplicates results, preferring corrected-query hits. No reranking.
        """
        try:
            # Normalize original query
            original = (query or "").strip()
            # 1) Spell-correct query via LangChain ChatCohere (best-effort)
            corrected = original
            try:
                from langchain_cohere import ChatCohere
                from langchain_core.messages import SystemMessage, HumanMessage
                # Acquire best-available chat key (and index) from KeyBank per call
                _chat_key, _chat_idx = self._keybank.get_key_with_index("rag_spell_correct")
                if _chat_key:
                    system_instr = (
                        "Correct minor spelling and typos in the user query for information retrieval. "
                        "Return only the corrected query as plain text with no quotes or extra words. "
                        "Do not introduce new named entities (people, places, orgs) that are not in the original. "
                        "Preserve names as typed unless the correction is clearly the same name with minor typos."
                    )

                    model_name = os.getenv("COHERE_CHAT_MODEL", "command-a-vision-07-2025")
                    msgs = [
                        SystemMessage(content=system_instr),
                        HumanMessage(content=f"Query: {original}"),
                    ]
                    # Retry without per-step timeouts; global timeout enforced by caller (frontend)
                    max_tries = self._keybank.key_count()
                    last_err = None
                    for _ in range(max_tries):
                        # Fetch a fresh key and rebuild client each attempt so we can rotate on failures
                        _chat_key, _chat_idx = self._keybank.get_key_with_index("rag_spell_correct")
                        chat = ChatCohere(model=model_name, cohere_api_key=_chat_key, temperature=0, max_tokens=64)
                        try:
                            # Log Cohere chat usage for spell-correction (CHAT key) with current key index
                            try:
                                api_logger = getattr(self, "api_logger", None)
                                if callable(api_logger):
                                    api_logger("chat", "COHERE_API_KEY_CHAT", note="rag_spell_correct", key_index=_chat_idx)
                            except Exception:
                                pass
                            resp_msg = chat.invoke(msgs)
                            txt = (getattr(resp_msg, "content", None) or "").strip()
                            if txt:
                                corrected = self._extract_single_query(txt, original)
                            break
                        except Exception as e:
                            last_err = e
                            try:
                                print(f"[CHAT][ERROR] note=rag_spell_correct key_index={_chat_idx} error={e}", flush=True)
                            except Exception:
                                pass
                            try:
                                self._keybank.penalize_key(_chat_idx, seconds=1.5)
                            except Exception:
                                pass
                            continue
            except Exception:
                # Best-effort; ignore spell-correction errors
                pass

            # Safety guard: only accept corrected if it's close to the original
            try:
                corr_norm = (corrected or "").strip()
                orig_tokens_raw = original.split()
                corr_tokens_raw = corr_norm.split()
                orig_lc = {t.lower().strip("'\".,!?():;[]{}") for t in orig_tokens_raw}
                corr_lc = [(t, t.lower().strip("'\".,!?():;[]{}")) for t in corr_tokens_raw]
                introduced_proper = any(self._is_proper_like(t) and lc not in orig_lc for t, lc in corr_lc)
                if introduced_proper:
                    corr_norm = original
                # Similarity metrics
                ratio = difflib.SequenceMatcher(None, original.lower(), corr_norm.lower()).ratio() if original and corr_norm else 0.0
                orig_tokens = set(original.lower().split())
                corr_tokens = set(corr_norm.lower().split())
                jaccard = (len(orig_tokens & corr_tokens) / max(1, len(orig_tokens | corr_tokens))) if orig_tokens or corr_tokens else 1.0
                # Thresholds: conservative
                if ratio < 0.70 and jaccard < 0.50:
                    corr_norm = original
                corrected = corr_norm
            except Exception:
                corrected = original

            # 2) Build query list: corrected first (preferred), then original if different
            queries = [corrected] if corrected == original else [corrected, original]
            print(f"[RAG] Using queries: {queries}")
            # Determine overfetch amount with a safe cap to avoid HNSW contiguous array issues
            try:
                collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
                count = int(collection.count())
            except Exception:
                count = None
            # Read caps from env with sensible defaults
            try:
                default_fetch = int(os.getenv("OFERGPT_RAG_FETCH_K", "500"))
            except Exception:
                default_fetch = 500
            try:
                max_fetch_cap = int(os.getenv("OFERGPT_RAG_MAX_FETCH", "1000"))
            except Exception:
                max_fetch_cap = 1000
            # Start from count or default, then cap and ensure at least k
            base = count if isinstance(count, int) and count >= 0 else default_fetch
            fetch_k = min(base, max_fetch_cap)
            try:
                fetch_k = max(fetch_k, int(k))
            except Exception:
                pass
            # Log effective k for this prompt
            try:
                print(
                    f"[RAG] Effective fetch_k={fetch_k} (requested k={k}, collection_count={count}, max_cap={max_fetch_cap})",
                    flush=True,
                )
            except Exception:
                pass
            # 3) Perform searches and merge results with preference for corrected pass
            merged: Dict[str, Dict[str, Any]] = {}

            
            
            for idx, q in enumerate(queries):
                try:
                    # Rotate query-time embedding key per call and log key index
                    _emb_key, _emb_idx = self._keybank.get_key_with_index("embed_query")
                    try:
                        from langchain_cohere import CohereEmbeddings as _CE
                        self.vectorstore_query._embedding_function = _CE(
                            model=self.embedding_model_name,
                            cohere_api_key=_emb_key,
                        )
                    except Exception:
                        pass
                    api_logger = getattr(self, "api_logger", None)
                    if callable(api_logger):
                        api_logger("embed", "COHERE_API_KEY_CHAT", note="rag_similarity_search", key_index=_emb_idx)
                    results = self.vectorstore_query.similarity_search_with_score(q, k=fetch_k)
                except Exception as inner_e:
                    print(f"[RAG] Search failed for query variant {idx}: {inner_e}")
                    try:
                        self._keybank.penalize_key(_emb_idx, seconds=1.5)
                    except Exception:
                        pass
                    # Retry once with a smaller k if possible
                    try:
                        smaller_k = max(int(k), min(200, fetch_k // 2))
                    except Exception:
                        smaller_k = int(k) if isinstance(k, int) else 5
                    if smaller_k < fetch_k:
                        try:
                            print(f"[RAG] Retrying search with smaller k={smaller_k}")
                            # Rotate a fresh key for retry as well
                            _emb_key2, _emb_idx2 = self._keybank.get_key_with_index("embed_query_retry")
                            try:
                                from langchain_cohere import CohereEmbeddings as _CE
                                self.vectorstore_query._embedding_function = _CE(
                                    model=self.embedding_model_name,
                                    cohere_api_key=_emb_key2,
                                )
                            except Exception:
                                pass
                            api_logger = getattr(self, "api_logger", None)
                            if callable(api_logger):
                                api_logger("embed", "COHERE_API_KEY_CHAT", note="rag_similarity_retry", key_index=_emb_idx2)
                            results = self.vectorstore_query.similarity_search_with_score(q, k=smaller_k)
                        except Exception as inner_e2:
                            print(f"[RAG] Retry failed for query variant {idx}: {inner_e2}")
                            try:
                                self._keybank.penalize_key(_emb_idx2, seconds=1.5)
                            except Exception:
                                pass
                            continue
                    else:
                        continue

                for doc, score in results:
                    key = self._doc_key(doc)
                    # Chroma returns a distance. Convert to cosine similarity where higher is better.
                    try:
                        distance = float(score)
                    except Exception:
                        distance = None
                    cosine_sim = (1.0 - distance) if distance is not None else None
                    entry = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "distance": distance,                 # raw distance from Chroma
                        "cosine_similarity": cosine_sim,      # derived similarity (higher is better)
                    }
                    if key in merged:
                        # Keep the better item by cosine similarity
                        existing = merged[key]
                        existing_sim = existing.get("cosine_similarity", float("-inf"))
                        new_sim = entry.get("cosine_similarity", float("-inf"))
                        if new_sim > existing_sim:
                            existing.update(entry)
                    else:
                        merged[key] = entry

            # 4) Sort by DESC cosine similarity only (no preference for corrected query)
            merged_list = list(merged.values())
            merged_list.sort(
                key=lambda x: (
                    -(x.get("cosine_similarity", float("-inf")) if x.get("cosine_similarity") is not None else float("-inf"))
                )
            )

            # 5) Trim to top-k
            final = []
            for item in merged_list[:k]:
                final.append(item)

            return final
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    

    ### Collection info ###

    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection."""
        try:
            collection = self.chroma_client.get_or_create_collection("ofergpt_memories")
            count = collection.count()
            result = collection.get()
            metadatas = result.get('metadatas', []) if result else []

            # Count unique photos, pdfs, and locations
            photo_filenames = set()
            pdf_filenames = set()
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
                elif t == 'location':
                    num_locations += 1

            num_photos = len(photo_filenames)
            num_pdfs = len(pdf_filenames)
            total_docs = num_photos + num_pdfs + num_locations

            return {
                "total_rag_chunks": count,
                "num_photos": num_photos,
                "num_pdfs": num_pdfs,
                "num_locations": num_locations,
                "total_docs": total_docs,
                "collection_name": "ofergpt_memories",
                "embedding_model": getattr(self, "embedding_model_name", "unknown")
            }
        except Exception as e:
            return {"error": str(e)}

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
            self.vectorstore_query = Chroma(
                client=self.chroma_client,
                collection_name="ofergpt_memories",
                embedding_function=self.query_embeddings,
            )
            print("✅ Empty collection re-created", flush=True)
        except Exception as e:
            print(f"Error clearing vector store: {e}", flush=True)

