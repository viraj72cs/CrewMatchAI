"""
In-memory session store for CrewMatch AI conversations.

Each session tracks the Gemini interaction chain, the created event,
and the recommended team so that confirmation → booking works correctly
across multiple chat turns.

Production notes:
  - TTL-based auto-expiration prevents unbounded RAM growth.
  - Max session cap (MAX_SESSIONS) ensures hard memory ceiling.
  - For multi-worker deployments (uvicorn --workers N), replace this
    module with a Redis-backed implementation.
"""

import uuid
import time
import threading
from typing import Optional


# ============================================================
# CONFIGURATION
# ============================================================

SESSION_TTL_SECONDS = 24 * 60 * 60   # 24 hours
MAX_SESSIONS = 1000                   # Hard cap on concurrent sessions
CLEANUP_INTERVAL = 300                # Run cleanup every 5 minutes


# ============================================================
# SESSION STORAGE
# ============================================================

# Keyed by session_id string.
# Each value is a dict with session data + "_last_accessed" timestamp.
_sessions: dict[str, dict] = {}
_lock = threading.Lock()


def _cleanup_expired() -> int:
    """Remove expired sessions. Returns count of removed sessions."""
    now = time.time()
    expired_ids = [
        sid for sid, s in _sessions.items()
        if (now - s.get("_last_accessed", 0)) > SESSION_TTL_SECONDS
    ]
    for sid in expired_ids:
        del _sessions[sid]
    if expired_ids:
        print(f"[SessionStore] Cleaned up {len(expired_ids)} expired sessions. Active: {len(_sessions)}")
    return len(expired_ids)


def _evict_oldest_if_full() -> None:
    """If we've hit MAX_SESSIONS, evict the oldest session."""
    while len(_sessions) >= MAX_SESSIONS:
        oldest_sid = min(
            _sessions,
            key=lambda sid: _sessions[sid].get("_last_accessed", 0)
        )
        del _sessions[oldest_sid]
        print(f"[SessionStore] Evicted oldest session {oldest_sid}. Active: {len(_sessions)}")


def _touch(session: dict) -> None:
    """Update the last-accessed timestamp on a session."""
    session["_last_accessed"] = time.time()


# ============================================================
# PUBLIC API
# ============================================================

def create_session() -> str:
    """Create a new blank session and return its ID."""

    with _lock:
        _cleanup_expired()
        _evict_oldest_if_full()

        session_id = str(uuid.uuid4())

        _sessions[session_id] = {
            "session_id": session_id,

            # Gemini Interactions API: previous interaction ID for conversation continuity.
            "interaction_id": None,

            # Set after create_event() succeeds.
            "event_id": None,

            # Event details stored for reference.
            "event_details": None,

            # List of dicts set when Gemini calls submit_recommendation.
            # Each dict: {crew_id, name, role, hourly_rate, estimated_cost, reason}
            "recommended_team": [],

            # Total cost from Gemini's recommendation.
            "total_cost": 0,

            # Budget from the organizer's event.
            "budget": 0,

            # True once Gemini has formally submitted a recommendation via tool call.
            # Cleared after booking completes.
            "awaiting_confirmation": False,

            # Booking result stored after team is booked.
            "booking_result": None,

            # Internal: TTL tracking
            "_last_accessed": time.time(),
            "_created_at": time.time(),
        }

    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """Return the session dict, or None if not found or expired."""
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return None

        # Check if expired
        if (time.time() - session.get("_last_accessed", 0)) > SESSION_TTL_SECONDS:
            del _sessions[session_id]
            return None

        _touch(session)
        return session


def get_or_create_session(session_id: Optional[str]) -> dict:
    """
    Return an existing session by ID, or create a new one.
    Always returns a valid session dict.
    """
    if session_id:
        session = get_session(session_id)
        if session:
            return session

    # Unknown or missing session ID → start fresh.
    new_id = create_session()
    return _sessions[new_id]


def update_session(session_id: str, **kwargs) -> None:
    """Update fields on an existing session."""
    with _lock:
        if session_id in _sessions:
            _sessions[session_id].update(kwargs)
            _touch(_sessions[session_id])


def clear_confirmation(session_id: str) -> None:
    """Clear the awaiting_confirmation flag after booking completes."""
    with _lock:
        if session_id in _sessions:
            _sessions[session_id]["awaiting_confirmation"] = False
            _sessions[session_id]["recommended_team"] = []
            _touch(_sessions[session_id])


def get_active_session_count() -> int:
    """Return the number of active (non-expired) sessions."""
    with _lock:
        _cleanup_expired()
        return len(_sessions)
