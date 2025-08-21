import os
import time
from typing import Any, Dict, Optional

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except Exception:  # Library may not be installed yet; handle gracefully
    MongoClient = None  # type: ignore
    class PyMongoError(Exception):
        pass
    try:
        print("[MONGO][INIT] pymongo not installed; Mongo logging disabled", flush=True)
    except Exception:
        pass


class MongoLogger:
    """Lightweight MongoDB logger for chat interactions.

    Safe no-op if MONGODB_URI is not set or pymongo is unavailable.
    """

    def __init__(self):
        self._uri = os.getenv("MONGODB_URI") or os.getenv("MONGODB_ATLAS_URI")
        self._db_name = os.getenv("OFERGPT_MONGO_DB", "bambi")
        self._coll_name = os.getenv("OFERGPT_MONGO_COLLECTION", "chat_logs")
        self._sess_coll_name = os.getenv("OFERGPT_MONGO_SESS_COLLECTION", "chat_sessions")
        self._client = None
        self._coll = None
        if not self._uri:
            try:
                print("[MONGO][INIT] disabled: missing MONGODB_URI", flush=True)
            except Exception:
                pass
            return
        if MongoClient is None:
            # Already logged above, but reiterate intent
            try:
                print("[MONGO][INIT] disabled: pymongo unavailable", flush=True)
            except Exception:
                pass
            return
        try:
            print(f"[MONGO][INIT] attempting connection db={self._db_name} coll={self._coll_name}", flush=True)
            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=3000)
            # Touch server to validate quickly
            try:
                self._client.admin.command("ping")
            except Exception:
                pass  # ping best-effort
            db = self._client[self._db_name]
            self._coll = db[self._coll_name]
            self._sess_coll = db[self._sess_coll_name]
            print(f"[MONGO][INIT] connected db={self._db_name} coll={self._coll_name} sess_coll={self._sess_coll_name}", flush=True)
        except Exception as e:
            # If connection fails at init, keep no-op mode
            self._client = None
            self._coll = None
            try:
                print(f"[MONGO][INIT][ERROR] connection_failed error={e}", flush=True)
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        return self._coll is not None

    def safe_log(
        self,
        *,
        user_message: str,
        ai_response: str,
        intent: Optional[str] = None,
        used_rag: Optional[bool] = None,
        rag_context_preview: Optional[str] = None,
        api_call_breakdown: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a chat log document; swallow all errors."""
        if not self.enabled:
            try:
                print("[MONGO][WRITE] skipped: logger disabled", flush=True)
            except Exception:
                pass
            return
        try:
            doc: Dict[str, Any] = {
                "ts": int(time.time()),
                "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "user_message": user_message,
                "ai_response": ai_response,
            }
            if intent is not None:
                doc["intent"] = intent
            if used_rag is not None:
                doc["used_rag"] = used_rag
            if rag_context_preview is not None:
                doc["rag_context_preview"] = rag_context_preview
            if api_call_breakdown:
                doc["api_calls"] = api_call_breakdown
            if metadata:
                doc["meta"] = metadata
            # Brief preview for logs without sensitive content
            try:
                preview = {
                    "user_len": len(user_message or ""),
                    "ai_len": len(ai_response or ""),
                    "intent": intent,
                    "used_rag": used_rag,
                }
                print(f"[MONGO][WRITE] inserting keys={list(doc.keys())} preview={preview}", flush=True)
            except Exception:
                pass
            res = self._coll.insert_one(doc)
            try:
                print(f"[MONGO][WRITE] success inserted_id={getattr(res, 'inserted_id', None)}", flush=True)
            except Exception:
                pass
        except (PyMongoError, Exception) as e:
            # Never raise from logger
            try:
                print(f"[MONGO][WRITE][ERROR] {e}", flush=True)
            except Exception:
                pass

    # --- Conversation session logging ---
    def upsert_session_turn(
        self,
        *,
        session_id: str,
        turn: Dict[str, Any],
        summary: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            try:
                print("[MONGO][SESS] skipped: logger disabled", flush=True)
            except Exception:
                pass
            return
        try:
            now = int(time.time())
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            set_on_insert = {
                "session_id": session_id,
                "started_ts": now,
                "started_iso": now_iso,
            }
            update: Dict[str, Any] = {
                "$setOnInsert": set_on_insert,
                "$push": {"turns": turn},
                "$set": {"updated_ts": now, "updated_iso": now_iso},
                "$inc": {"turns_count": 1},
            }
            if summary is not None:
                update["$set"]["summary"] = summary
            if meta is not None:
                update["$set"]["meta"] = meta
            try:
                print(f"[MONGO][SESS] upsert turn session_id={session_id} keys={list(turn.keys())}", flush=True)
            except Exception:
                pass
            res = self._sess_coll.update_one({"session_id": session_id}, update, upsert=True)
            try:
                print(f"[MONGO][SESS] upsert result matched={res.matched_count} modified={res.modified_count} upserted_id={getattr(res, 'upserted_id', None)}", flush=True)
            except Exception:
                pass
        except (PyMongoError, Exception) as e:
            try:
                print(f"[MONGO][SESS][ERROR] {e}", flush=True)
            except Exception:
                pass

    def end_session(self, *, session_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            try:
                print("[MONGO][SESS] end skipped: logger disabled", flush=True)
            except Exception:
                pass
            return
        try:
            now = int(time.time())
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            update: Dict[str, Any] = {"$set": {"ended_ts": now, "ended_iso": now_iso, "ended": True}}
            if meta is not None:
                update["$set"]["meta"] = meta
            res = self._sess_coll.update_one({"session_id": session_id}, update, upsert=False)
            try:
                print(f"[MONGO][SESS] end result matched={res.matched_count} modified={res.modified_count}", flush=True)
            except Exception:
                pass
        except (PyMongoError, Exception) as e:
            try:
                print(f"[MONGO][SESS][END][ERROR] {e}", flush=True)
            except Exception:
                pass


# Simple accessor to avoid multiple clients
_singleton: Optional[MongoLogger] = None

def get_mongo_logger() -> MongoLogger:
    global _singleton
    if _singleton is None:
        _singleton = MongoLogger()
    return _singleton
