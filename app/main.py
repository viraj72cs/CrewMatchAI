import os
import sys

# Ensure Windows PowerShell / CMD stdout supports UTF-8 emojis without charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.services.gemini import chat_with_gemini
from app.services.session_store import get_or_create_session, clear_confirmation

from app.tools.booking_tools import book_team
from app.tools.cancellation_tools import cancel_booking, find_replacement_candidates
from app.models.schemas import CancellationRequest, ReplacementRequest

from app.config import DEBUG_MODE, CORS_ORIGINS

app = FastAPI(
    title="CrewMatch AI Backend",
    version="1.0.0"
)

# Configured via CORS_ORIGINS env var (defaults to ["*"] in dev).
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model_name: Optional[str] = "gemini-2.5-flash"


# ============================================================
# CONFIRMATION DETECTION
# Simple keyword check — sufficient for hackathon demo.
# ============================================================

CONFIRMATION_KEYWORDS = [
    "yes", "book", "confirm", "go ahead", "proceed",
    "book them", "looks good", "do it", "ok", "okay",
    "book the team", "book all", "yes please", "haan",
    "haan book karo", "book kar do"
]


def _is_confirmation(message: str) -> bool:
    lower = message.lower().strip()
    return any(kw in lower for kw in CONFIRMATION_KEYWORDS)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "app": "CrewMatch AI",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/models")
def get_models():
    """Return available LLM models for dynamic model selection."""
    from app.services.gemini import AVAILABLE_MODELS
    return {
        "success": True,
        "models": AVAILABLE_MODELS
    }


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Main AI chat endpoint. Maintains multi-turn conversation state
    via session_id. Returns structured response for Flutter.

    Response shape:
    {
        "success": bool,
        "session_id": str,
        "state": "chatting" | "recommending" | "awaiting_confirmation"
                 | "booking_complete" | "booking_partial",
        "message": str,
        "recommendation": dict | null,
        "booking_result": dict | null
    }
    """

    session = get_or_create_session(request.session_id)
    session_id = session["session_id"]

    # ----------------------------------------------------------
    # CONFIRMATION SHORTCUT
    # If the session is awaiting confirmation and the user says yes,
    # skip Gemini and book the team directly.
    # ----------------------------------------------------------

    if session.get("awaiting_confirmation") and _is_confirmation(request.message):

        event_id = session.get("event_id")
        team = session.get("recommended_team", [])

        if not event_id:
            return {
                "success": False,
                "session_id": session_id,
                "state": "error",
                "message": (
                    "I don't have an event ID yet. "
                    "Please confirm the event details first "
                    "so I can create the event before booking."
                ),
                "recommendation": None,
                "booking_result": None
            }

        if not team:
            return {
                "success": False,
                "session_id": session_id,
                "state": "error",
                "message": (
                    "No recommended team found in session. "
                    "Please ask me to recommend a crew first."
                ),
                "recommendation": None,
                "booking_result": None
            }

        booking_result = book_team(
            event_id=event_id,
            team=team
        )

        clear_confirmation(session_id)

        state = (
            "booking_complete"
            if booking_result["total_failed"] == 0
            else "booking_partial"
        )

        if state == "booking_complete":
            message = (
                f"🎉 Your event crew has been booked successfully! "
                f"{booking_result['total_booked']} crew members confirmed."
            )
        else:
            message = (
                f"⚠️ Partially booked: "
                f"{booking_result['total_booked']} succeeded, "
                f"{booking_result['total_failed']} failed. "
                f"Please review the failures."
            )

        return {
            "success": booking_result["success"],
            "session_id": session_id,
            "state": state,
            "message": message,
            "recommendation": None,
            "booking_result": booking_result
        }

    # ----------------------------------------------------------
    # STANDARD GEMINI CHAT
    # ----------------------------------------------------------

    result = chat_with_gemini(
        message=request.message,
        session=session,
        model_name=request.model_name
    )

    response_text = result["response_text"]
    recommendation = result.get("recommendation")
    booking_triggered = result.get("booking_triggered", False)

    # Determine state for Flutter.
    if booking_triggered:
        state = "booking_complete"
    elif session.get("awaiting_confirmation"):
        state = "awaiting_confirmation"
    elif recommendation:
        state = "recommending"
    else:
        state = "chatting"

    return {
        "success": True,
        "session_id": session_id,
        "state": state,
        "message": response_text,
        "recommendation": recommendation,
        "booking_result": None
    }


# ============================================================
# MANUAL BOOKING ENDPOINT
# Allows Flutter to directly trigger a team booking using
# stored session data without a Gemini chat turn.
# ============================================================

@app.post("/book-team/{session_id}")
def book_team_endpoint(session_id: str):
    """
    Directly book the team stored in the session.
    Call this after the user confirms in Flutter UI.
    """
    from app.services.session_store import get_session

    session = get_session(session_id)
    if not session:
        return {
            "success": False,
            "error": f"Session {session_id} not found."
        }

    event_id = session.get("event_id")
    team = session.get("recommended_team", [])

    if not event_id:
        return {
            "success": False,
            "error": "No event_id in session."
        }

    if not team:
        return {
            "success": False,
            "error": "No recommended team in session."
        }

    result = book_team(event_id=event_id, team=team)
    clear_confirmation(session_id)
    return result



@app.post("/cancel")
def cancel(request: CancellationRequest):
    """Cancel a booking by setting its status to CANCELLED."""
    result = cancel_booking(booking_id=request.booking_id)
    return result


@app.post("/replace")
def replace(request: ReplacementRequest):
    """
    Find replacement candidates for a role, excluding already-booked
    and cancelled crew members.
    """
    result = find_replacement_candidates(
        role=request.role,
        event_id=request.event_id,
        event_date=request.event_date,
        start_datetime=request.start_datetime,
        end_datetime=request.end_datetime,
        budget=request.budget,
        exclude_crew_ids=request.exclude_crew_ids
    )
    return {
        "success": True,
        "data": result
    }


# ============================================================
# DEBUG ENDPOINTS (only registered when DEBUG_MODE=true)
# ============================================================

if DEBUG_MODE:
    from app.tools.planner_tools import find_crew_candidates

    @app.get("/test-planner")
    def test_planner(
            role: str,
            event_date: str,
            start_datetime: str,
            end_datetime: str,
            budget: float
    ):
        result = find_crew_candidates(
            role=role,
            event_date=event_date,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            budget=budget
        )
        return {
            "success": True,
            "data": result
        }

    @app.get("/session/{session_id}")
    def get_session_state(session_id: str):
        """Debug: inspect the current session state."""
        from app.services.session_store import get_session

        session = get_session(session_id)
        if not session:
            return {
                "success": False,
                "error": f"Session {session_id} not found."
            }

        return {
            "success": True,
            "session_id": session_id,
            "event_id": session.get("event_id"),
            "awaiting_confirmation": session.get("awaiting_confirmation"),
            "team_count": len(session.get("recommended_team", [])),
            "recommended_team": session.get("recommended_team", []),
            "total_cost": session.get("total_cost"),
            "budget": session.get("budget"),
        }