import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

from .session_store import SessionStore


class ChatPersistence:
    """Fire-and-forget persistence for chat messages and threads.

    Uses a small thread pool to avoid blocking Streamlit render while saving
    to Upstash Redis. Safe to call from the UI layer.
    """

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="persist")
        self._local = threading.local()

    def _get_store(self) -> Optional[SessionStore]:
        try:
            # Create one SessionStore per worker thread to avoid cross-thread issues
            store = getattr(self._local, "store", None)
            if store is None:
                store = SessionStore()
                self._local.store = store
            return store
        except Exception as e:
            print(f"[Persist] store init error: {e}", flush=True)
            return None

    def ensure_thread(self, session_id: str, thread_id: Optional[str], default_name: str = "Chat 1") -> Optional[str]:
        """Ensure there is a valid thread id. If missing, create one synchronously.
        Returns the resolved thread_id or None if store unavailable.
        """
        store = self._get_store()
        if not store or not session_id:
            return thread_id
        if thread_id:
            return thread_id
        try:
            t = store.create_thread(session_id, name=default_name)
            return t.get("id")
        except Exception as e:
            print(f"[Persist] ensure_thread error: {e}", flush=True)
            return thread_id

    def save_message_async(self, session_id: str, thread_id: Optional[str], role: str, content: str) -> None:
        """Queue message save; returns immediately."""
        if not session_id or not thread_id or not content:
            return
        self._executor.submit(self._save_message, session_id, thread_id, role, content)

    # ----- Internal worker function -----
    def _save_message(self, session_id: str, thread_id: str, role: str, content: str) -> None:
        try:
            store = self._get_store()
            if not store:
                return
            store.append_message(session_id, thread_id, role, content)
        except Exception as e:
            print(f"[Persist] save_message error: {e}", flush=True)


@lru_cache(maxsize=1)
def get_persistence() -> ChatPersistence:
    return ChatPersistence(max_workers=4)
