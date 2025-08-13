"""
Compare similarity between a stored embedding in ChromaDB and an on-demand query embedding.

Usage examples:
  python dev_tools/compare_embeddings.py --id SOME_DOC_ID --query "your query text"
  python dev_tools/compare_embeddings.py --id SOME_DOC_ID --query @-   # then type query and press Ctrl+Z (Windows) / Ctrl+D (Unix)

Notes:
- Uses the same settings as RAGSystem:
  - Chroma PersistentClient at ./data/embeddings
  - Collection name: "ofergpt_memories"
  - Cohere embeddings model: embed-english-v3.0
- Requires env var COHERE_API_KEY_EMBED
"""

import argparse
import json
import os
import sys
import math
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import CohereEmbeddings
import cohere


COLLECTION_NAME = "ofergpt_memories"
DEFAULT_DB_PATH = "./data/embeddings"
EMBED_MODEL = "embed-english-v3.0"


def _read_stdin_fallback() -> str:
    """Allow --query @- to read from stdin."""
    return sys.stdin.read()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        raise ValueError("Vectors must be same non-zero length")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def load_query_embedding(text: str, api_key: str) -> List[float]:
    emb = CohereEmbeddings(model=EMBED_MODEL, cohere_api_key=api_key)
    # For Cohere v3 embeddings, use embed_query for queries and embed_documents for documents.
    # This matches the directional retrieval setup used by LangChain vectorstores.
    vec = emb.embed_query(text)
    return vec


def get_stored_embedding(client_path: str, doc_id: str) -> Optional[List[float]]:
    client = chromadb.PersistentClient(path=client_path, settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(COLLECTION_NAME)
    res = col.get(ids=[doc_id], include=["embeddings", "documents", "metadatas"])  # type: ignore[arg-type]
    if not res or not res.get("ids"):
        return None
    if not res.get("embeddings"):
        # This should not happen in normal Chroma usage; embeddings are stored.
        return None
    # res["embeddings"] is List[List[float]] aligned with res["ids"]
    return res["embeddings"][0]


def search_topk(
    client_path: str,
    query_text: str,
    api_key: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Mirror chatbot retrieval: embed query, query Chroma for distances, convert to cosine_similarity.
    Optionally apply Cohere Rerank if OFERGPT_RERANK=1.
    """
    # 1) Build query embedding (directional)
    emb = CohereEmbeddings(model=EMBED_MODEL, cohere_api_key=api_key)
    qvec = emb.embed_query(query_text)

    # 2) Query Chroma directly for top-k by distance
    client = chromadb.PersistentClient(path=client_path, settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(COLLECTION_NAME)
    qres = col.query(query_embeddings=[qvec], n_results=k, include=["documents", "metadatas", "distances"])  # type: ignore[arg-type]

    docs = []
    if qres and qres.get("ids"):
        ids = qres.get("ids", [[]])[0] or []
        docs_list = qres.get("documents", [[]])[0] or []
        metas_list = qres.get("metadatas", [[]])[0] or []
        dists = qres.get("distances", [[]])[0] or []

        # Pad lists to the same length as ids
        n = len(ids)
        if len(docs_list) < n:
            docs_list = docs_list + [""] * (n - len(docs_list))
        if len(metas_list) < n:
            metas_list = metas_list + [{} for _ in range(n - len(metas_list))]
        if len(dists) < n:
            dists = dists + [None] * (n - len(dists))

        for i, did in enumerate(ids):
            dist = dists[i]
            try:
                distance = float(dist) if dist is not None else None
            except Exception:
                distance = None
            cosine_sim = (1.0 - distance) if distance is not None else None
            docs.append({
                "id": did,
                "content": docs_list[i],
                "metadata": metas_list[i] or {},
                "distance": distance,
                "cosine_similarity": cosine_sim,
            })

    # 3) Optional Cohere Rerank over the head documents (content-only), to mirror chatbot
    try:
        if os.getenv("OFERGPT_RERANK", "0") == "1" and docs:
            try:
                rr_top_k = int(os.getenv("OFERGPT_RERANK_TOP_K", os.getenv("OFERGPT_RAG_TOP_K", "20")))
            except Exception:
                rr_top_k = 20
            head = docs[:rr_top_k]
            candidates = [d.get("content", "") or "" for d in head]
            if any(candidates):
                rerank_model = os.getenv("OFERGPT_RERANK_MODEL", "rerank-english-v3.0")
                ch = cohere.Client(os.getenv("COHERE_API_KEY_CHAT"))
                rr = ch.rerank(model=rerank_model, query=query_text, documents=candidates, top_n=len(candidates))
                results = getattr(rr, "results", None) or []
                ordered = sorted(
                    [(r.index, getattr(r, "relevance_score", 0.0)) for r in results],
                    key=lambda x: x[1], reverse=True,
                )
                reordered_head = [head[idx] for idx, _ in ordered if 0 <= idx < len(head)]
                tail = docs[len(head):]
                docs = reordered_head + tail
    except Exception as _e:
        # Best-effort; ignore rerank errors
        pass

    return docs


def main():
    parser = argparse.ArgumentParser(description="Compare similarity between a stored Chroma embedding and a query embedding, or run a top-k search like the chatbot.")
    parser.add_argument("--id", required=True, help="Document ID in Chroma collection ofergpt_memories")
    parser.add_argument("--query", required=True, help="Query text, or @- to read from stdin")
    parser.add_argument("--embeddings-dir", default=DEFAULT_DB_PATH, help="Path to Chroma persistent dir (default: ./data/embeddings)")
    parser.add_argument("--print-meta", action="store_true", help="Print stored document metadata for the id")
    parser.add_argument("--search-topk", type=int, default=0, help="If >0, also run a top-k search and print the ranked hits (mirrors chatbot retrieval)")

    args = parser.parse_args()

    cohere_key = os.getenv("COHERE_API_KEY_EMBED")
    if not cohere_key:
        print("ERROR: COHERE_API_KEY_EMBED is not set in environment.", file=sys.stderr)
        sys.exit(2)

    query_text = _read_stdin_fallback() if args.query == "@-" else args.query

    # 1) Load stored embedding
    stored_vec = get_stored_embedding(args.embeddings_dir, args.id)
    if stored_vec is None:
        print(f"ERROR: No embedding found for id: {args.id}")
        sys.exit(1)

    # Optionally print metadata
    if args.print_meta:
        client = chromadb.PersistentClient(path=args.embeddings_dir, settings=Settings(anonymized_telemetry=False))
        col = client.get_or_create_collection(COLLECTION_NAME)
        res = col.get(ids=[args.id], include=["documents", "metadatas"])  # type: ignore[arg-type]
        meta = res.get("metadatas", [{}])[0]
        doc = res.get("documents", [""])[0]
        print("-- Stored Item --")
        print(json.dumps({
            "id": args.id,
            "metadata": meta,
            "document": doc,
        }, ensure_ascii=False, indent=2))
        print()

    # 2) Build query embedding
    query_vec = load_query_embedding(query_text, cohere_key)

    # 3) Cosine similarity
    sim = cosine_similarity(stored_vec, query_vec)

    print(json.dumps({
        "id": args.id,
        "query": query_text,
        "similarity_cosine": sim,
    }, ensure_ascii=False, indent=2))
    # Brief diagnostic to confirm embedding modes
    print("note: compared stored document embedding (embed_documents) to query embedding (embed_query)")

    # 4) Optional: run a real search like the chatbot and print top-k
    if args.search_topk and args.search_topk > 0:
        hits = search_topk(args.embeddings_dir, query_text, cohere_key, k=args.search_topk)
        print()
        print(f"-- Top {args.search_topk} by vector distance (cosine_similarity = 1 - distance); rerank={'on' if os.getenv('OFERGPT_RERANK','0')=='1' else 'off'} --")
        for i, h in enumerate(hits, start=1):
            meta = h.get("metadata") or {}
            # Construct a stable-ish key similar to RAGSystem's dedupe (best-effort)
            key = meta.get("idempotency_key") or meta.get("filename") or h.get("id")
            print(json.dumps({
                "rank": i,
                "id": h.get("id"),
                "key": key,
                "cosine_similarity": h.get("cosine_similarity"),
                "distance": h.get("distance"),
                "metadata": meta,
            }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
