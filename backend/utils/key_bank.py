import os
import time
import heapq
import threading
from typing import List, Optional, Tuple

try:
    import cohere  # type: ignore
except Exception:  # pragma: no cover
    cohere = None  # Will raise at runtime if used without package


class KeyBank:
    """
    A thread-safe key scheduler with per-key token buckets (default ~10 req/min per key).

    Strategy:
    - Maintain a heap of (available_at_epoch_seconds, index, key) for earliest-available selection across keys.
    - Each key has a token bucket with:
        capacity = per_key_rate_per_min (default derived from COHERE_KEY_RATE_PER_MIN or 60/COHERE_KEY_INTERVAL_SECONDS)
        refill_rate = capacity / 60 tokens per second
      This allows up to 'capacity' burst calls per minute per key without forced 6s sleeps after each call.

    Supports loading keys from environment via:
    - COHERE_API_KEY_CHAT: comma/semicolon/space-separated list
    - or numbered: COHERE_API_KEY_CHAT_1 .. COHERE_API_KEY_CHAT_20
    Env:
    - COHERE_KEY_RATE_PER_MIN: optional explicit per-key rate (e.g., 10). If not set, falls back to 60/COHERE_KEY_INTERVAL_SECONDS (default 6s -> 10/min).
    """

    def __init__(self, keys: List[str], per_key_interval_seconds: float = 6.0, per_key_rate_per_min: Optional[float] = None):
        if not keys:
            raise ValueError("KeyBank requires at least one Cohere API key")
        self._lock = threading.Lock()
        # Compute per-key bucket params
        if per_key_rate_per_min is not None and per_key_rate_per_min > 0:
            self._capacity_int = int(per_key_rate_per_min)
        else:
            # Back-compat: derive from interval (e.g., 6s -> 10/min)
            interval = max(0.1, float(per_key_interval_seconds))
            self._capacity_int = max(1, int(round(60.0 / interval)))
        self._rate = float(self._capacity_int) / 60.0  # tokens per second
        # Initialize per-key token state
        now = time.time()
        self._tokens: List[int] = [self._capacity_int for _ in keys]
        self._last: List[float] = [now for _ in keys]
        # Initialize heap with immediate availability for each key
        self._heap: List[Tuple[float, int, str]] = [(0.0, i, k) for i, k in enumerate(keys)]
        heapq.heapify(self._heap)

    @classmethod
    def from_env(cls) -> "KeyBank":
        keys: List[str] = []

        # 1) Multi-value COHERE_API_KEY_CHAT
        raw = os.getenv("COHERE_API_KEY_CHAT", "").strip()
        if raw:
            # split by common separators
            parts = [p.strip() for p in raw.replace(";", ",").replace(" ", ",").split(",")]
            keys.extend([p for p in parts if p])

        # 2) Numbered keys: COHERE_API_KEY_CHAT_1..20
        if not keys:
            for i in range(1, 21):
                v = os.getenv(f"COHERE_API_KEY_CHAT_{i}")
                if v and v.strip():
                    keys.append(v.strip())

        if not keys:
            raise EnvironmentError(
                "No Cohere keys found. Set COHERE_API_KEY_CHAT (comma-separated) or COHERE_API_KEY_CHAT_1..N"
            )

        # Allow override of per-key interval (seconds) via env, default 6s
        try:
            interval = float(os.getenv("COHERE_KEY_INTERVAL_SECONDS", "6"))
        except ValueError:
            interval = 6.0
        # Optional explicit per-key rate per minute
        try:
            rpm = os.getenv("COHERE_KEY_RATE_PER_MIN")
            rate_per_min = float(rpm) if rpm is not None and rpm.strip() != "" else None
            if rate_per_min is not None and rate_per_min <= 0:
                rate_per_min = None
        except Exception:
            rate_per_min = None

        return cls(keys=keys, per_key_interval_seconds=interval, per_key_rate_per_min=rate_per_min)

    def get_client(self) -> "cohere.Client":
        """Return a new cohere.Client using the best-available key.

        This call is blocking if all keys are temporarily rate-limited.
        """
        if cohere is None:
            raise RuntimeError("cohere package is not installed. Please `pip install cohere`. ")
        key = self.get_key("chat")
        return cohere.Client(key)

    def get_key(self, purpose: str = "chat") -> str:
        """Return the best-available key (string only)."""
        key, _ = self.get_key_with_index(purpose)
        return key

    def get_key_with_index(self, purpose: str = "chat") -> tuple[str, int]:
        """Return the best-available key and its stable index in the heap order.

        Token-bucket behavior per key:
        - If tokens >= 1: consume 1 and return immediately (no sleep).
        - Else: compute wait until 1 token is available, schedule, and sleep that duration.
        """
        with self._lock:
            available_at, idx, key = heapq.heappop(self._heap)
            now = time.time()
            # Respect heap availability (could be in the future due to prior scheduling)
            wait_heap = max(0.0, available_at - now)
            if wait_heap > 0:
                # While we wait, we also advance token refill baseline to 'available_at'
                now = available_at
            # Refill bucket for this key using whole tokens only
            delta = max(0.0, now - self._last[idx])
            if delta > 0:
                accrued = int(delta * self._rate)
                if accrued > 0:
                    self._tokens[idx] = min(self._capacity_int, self._tokens[idx] + accrued)
                    # Advance last by the exact time that produced whole tokens; keep fractional remainder in time
                    self._last[idx] = self._last[idx] + (accrued / self._rate)
            wait_extra = 0.0
            if self._tokens[idx] >= 1:
                # Consume immediately
                self._tokens[idx] -= 1
                # Next availability depends on remaining tokens
                if self._tokens[idx] >= 1:
                    next_available = now  # still burst capacity
                else:
                    # Next whole token completes at last + 1/rate
                    next_available = self._last[idx] + (1.0 / self._rate)
            else:
                # Need to wait for token to refill
                next_available = self._last[idx] + (1.0 / self._rate)
                wait_extra = max(0.0, next_available - now)
                # Reserve the upcoming token by advancing last to next_available (so the next refill starts from there)
                self._last[idx] = next_available
                # Push back immediately so others can choose among keys
            heapq.heappush(self._heap, (next_available, idx, key))

        total_wait = wait_heap + wait_extra
        if total_wait > 0:
            # Log the reason(s) for waiting
            try:
                if wait_extra > 0:
                    # Waiting specifically because this key had no whole tokens available
                    print(f"[KEYBANK_WAIT] reason=tokens_depleted key_index={idx} wait_seconds={wait_extra:.3f}", flush=True)
                if wait_heap > 0 and wait_extra == 0:
                    # Waiting because the heap scheduled this key in the future (all keys previously exhausted)
                    print(f"[KEYBANK_WAIT] reason=scheduled key_index={idx} wait_seconds={wait_heap:.3f}", flush=True)
            except Exception:
                pass
            time.sleep(total_wait)

        return key, idx

    def peek_best_time(self) -> float:
        """Return the epoch seconds when the best key becomes available (no mutation)."""
        with self._lock:
            next_time, _, _ = self._heap[0]
            return next_time

    def peek_best_key(self) -> str:
        """Return the current best key without incrementing usage (no mutation)."""
        with self._lock:
            _, _, key = self._heap[0]
            return key

    def key_count(self) -> int:
        """Return the number of keys managed by this bank."""
        # All internal arrays share the same length
        return len(self._last)

    def penalize_key(self, idx: int, seconds: float = 1.5) -> None:
        """Push the specified key's availability into the future by 'seconds'.

        Useful after a retryable API failure to steer the next attempt toward a different key.
        Complexity is O(N) over the heap size (N ≤ keys count), which is fine for small pools.
        """
        if seconds <= 0:
            return
        with self._lock:
            now = time.time()
            # Compute the current worst (latest) availability across all keys
            current_max_available = max((available_at for available_at, _, _ in self._heap), default=now)
            new_heap: List[Tuple[float, int, str]] = []
            target_found = False
            for available_at, i, k in self._heap:
                if i == idx and not target_found:
                    target_found = True
                    # Always push beyond the current maximum so this key is least preferred next
                    bumped = max(available_at, now, current_max_available) + seconds
                    new_heap.append((bumped, i, k))
                else:
                    new_heap.append((available_at, i, k))
            if not target_found and 0 <= idx < len(self._last):
                # If not present (shouldn't happen), reinsert with penalty using peeked key string
                key = None
                for _, i, k in self._heap:
                    if i == idx:
                        key = k
                        break
                if key is None:
                    # Fallback: derive from initial construction order
                    # This is a rare edge; do nothing if key unknown
                    pass
                else:
                    # Insert at worst availability
                    new_heap.append((max(now, current_max_available) + seconds, idx, key))
            heapq.heapify(new_heap)
            self._heap = new_heap
            try:
                print(f"[KEYBANK_PENALIZE] key_index={idx} seconds={seconds:.3f}", flush=True)
            except Exception:
                pass


# Singleton helper for convenience
_keybank_singleton: Optional[KeyBank] = None
_keybank_lock = threading.Lock()


def get_keybank() -> KeyBank:
    global _keybank_singleton
    if _keybank_singleton is None:
        with _keybank_lock:
            if _keybank_singleton is None:
                _keybank_singleton = KeyBank.from_env()
    return _keybank_singleton


def get_cohere_client() -> "cohere.Client":
    """Shortcut to fetch a client using the shared key bank."""
    return get_keybank().get_client()
