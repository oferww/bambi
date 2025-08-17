import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional

try:
    from upstash_redis import Redis
except Exception as e:  # pragma: no cover
    Redis = None  # type: ignore


class SessionStore:
    """Thin wrapper over Upstash Redis to manage user threads and messages.

    Keys layout:
      - user:{session_id}:threads -> JSON list of thread dicts [{id, name, created_at}]
      - thread:{session_id}:{thread_id}:messages -> JSON list of messages [{role, content, ts}]
    """

    def __init__(self):
        url = os.getenv("UPSTASH_REDIS_REST_URL")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        if not url or not token:
            raise RuntimeError("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set in environment")
        # Guard: users sometimes provide the Redis connection string (rediss://) instead of REST URL (https://)
        if url.startswith("redis://") or url.startswith("rediss://"):
            raise RuntimeError(
                "UPSTASH_REDIS_REST_URL appears to be a Redis connection string (rediss://). "
                "This app uses the Upstash REST client. Please set the REST URL (https://...) and REST TOKEN."
            )
        if Redis is None:
            raise RuntimeError("upstash-redis package not available. Please install 'upstash-redis'.")
        self.client = Redis(url=url, token=token)

    # ----- Threads -----
    def _threads_key(self, session_id: str) -> str:
        return f"user:{session_id}:threads"

    def list_threads(self, session_id: str) -> List[Dict[str, Any]]:
        raw = self.client.get(self._threads_key(session_id))
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:
            return []

    def create_thread(self, session_id: str, name: Optional[str] = None) -> Dict[str, Any]:
        thread_id = str(uuid.uuid4())
        thread = {
            "id": thread_id,
            "name": name or "New chat",
            "created_at": int(time.time()),
        }
        threads = self.list_threads(session_id)
        threads.insert(0, thread)  # newest first
        self.client.set(self._threads_key(session_id), json.dumps(threads))
        return thread

    # ----- Messages -----
    def _messages_key(self, session_id: str, thread_id: str) -> str:
        return f"thread:{session_id}:{thread_id}:messages"

    def get_messages(self, session_id: str, thread_id: str) -> List[Dict[str, Any]]:
        raw = self.client.get(self._messages_key(session_id, thread_id))
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:
            return []

    def save_messages(self, session_id: str, thread_id: str, messages: List[Dict[str, Any]]) -> None:
        self.client.set(self._messages_key(session_id, thread_id), json.dumps(messages))

    def append_message(self, session_id: str, thread_id: str, role: str, content: str) -> None:
        msgs = self.get_messages(session_id, thread_id)
        msgs.append({"role": role, "content": content, "ts": int(time.time())})
        self.save_messages(session_id, thread_id, msgs)
