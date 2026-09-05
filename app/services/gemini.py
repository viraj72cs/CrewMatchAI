import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from google import genai

from app.config import GEMINI_API_KEY

from app.tools.planner_tools import find_crew_candidates, find_team_candidates
from app.tools.event_tools import create_event
from app.tools.booking_tools import book_crew_member
from app.tools.cancellation_tools import cancel_booking, find_replacement_candidates


client = genai.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# TOOL 0: FIND TEAM CANDIDATES (MULTI-ROLE CONSOLIDATED)
# ============================================================

find_team_candidates_tool = {
    "type": "function",
    "name": "find_team_candidates",
    "description": (
        "Retrieve real CrewMatch crew candidates for MULTIPLE roles simultaneously in a single search. "
        "Returns crew profiles, skills, and availability from Supabase for all requested roles. "
        "ALWAYS PREFER THIS TOOL over find_crew_candidates when searching for crew members."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of required crew roles such as ['Photographer', 'Decorator']."
            },
            "event_date": {
                "type": "string",
                "description": "Event date in YYYY-MM-DD format."
            },
            "start_datetime": {
                "type": "string",
                "description": "Event start datetime in ISO format."
            },
            "end_datetime": {
                "type": "string",
                "description": "Event end datetime in ISO format."
            },
            "budget": {
                "type": "number",
                "description": "Budget available for the crew requirements."
            }
        },
        "required": [
            "roles",
            "event_date",
            "start_datetime",
            "end_datetime",
            "budget"
        ]
    }
}


# ============================================================
# TOOL 1: FIND CREW CANDIDATES
# ============================================================

find_crew_candidates_tool = {
    "type": "function",
    "name": "find_crew_candidates",
    "description": (
        "Retrieve real CrewMatch crew candidates for an event role. "
        "Returns crew profiles, skills, and availability from Supabase. "
        "Use this whenever the organizer needs crew recommendations. "
        "Never invent crew members."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "description": (
                    "Required crew role such as Photographer, "
                    "Videographer, Decorator, Security, Anchor, "
                    "or Sound Engineer."
                )
            },
            "event_date": {
                "type": "string",
                "description": "Event date in YYYY-MM-DD format."
            },
            "start_datetime": {
                "type": "string",
                "description": "Event start datetime in ISO format."
            },
            "end_datetime": {
                "type": "string",
                "description": "Event end datetime in ISO format."
            },
            "budget": {
                "type": "number",
                "description": "Budget available for this requirement."
            }
        },
        "required": [
            "role",
            "event_date",
            "start_datetime",
            "end_datetime",
            "budget"
        ]
    }
}


# ============================================================
# TOOL 2: CREATE EVENT
# ============================================================

create_event_tool = {
    "type": "function",
    "name": "create_event",
    "description": (
        "Create a CrewMatch event in Supabase after the organizer "
        "has clearly confirmed the event details. "
        "Returns the event ID needed for later booking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "event_type": {
                "type": "string"
            },
            "event_date": {
                "type": "string",
                "description": "Event date in YYYY-MM-DD format."
            },
            "start_time": {
                "type": "string",
                "description": "Event start time."
            },
            "end_time": {
                "type": "string",
                "description": "Event end time."
            },
            "location": {
                "type": "string"
            },
            "guest_count": {
                "type": "integer"
            },
            "total_budget": {
                "type": "number"
            }
        },
        "required": [
            "event_type",
            "event_date",
            "start_time",
            "end_time",
            "location",
            "guest_count",
            "total_budget"
        ]
    }
}


# ============================================================
# TOOL 3: BOOK CREW MEMBER
# ============================================================

book_crew_member_tool = {
    "type": "function",
    "name": "book_crew_member",
    "description": (
        "Create a confirmed booking for a specific CrewMatch crew "
        "member. Only use this after the organizer explicitly "
        "confirms that they want to book the recommended crew."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "Supabase event ID."
            },
            "crew_id": {
                "type": "string",
                "description": "Supabase crew member ID."
            },
            "role": {
                "type": "string",
                "description": "Crew member's event role."
            },
            "agreed_rate": {
                "type": "number",
                "description": "Agreed hourly rate."
            }
        },
        "required": [
            "event_id",
            "crew_id",
            "role",
            "agreed_rate"
        ]
    }
}


# ============================================================
# TOOL 4: SUBMIT RECOMMENDATION
# Gemini calls this to formally register its chosen team.
# Python saves the crew IDs so they can be booked on confirmation.
# This avoids parsing natural language for crew IDs.
# ============================================================

submit_recommendation_tool = {
    "type": "function",
    "name": "submit_recommendation",
    "description": (
        "After evaluating all candidates, call this tool once to "
        "formally submit the recommended team to the system. "
        "This registers the crew IDs so the organizer can confirm "
        "and book the team. Always call this after producing your "
        "team recommendation, before asking the organizer to confirm."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "recommended_team": {
                "type": "array",
                "description": (
                    "List of recommended crew members. "
                    "Each item must include crew_id, name, role, "
                    "hourly_rate, estimated_cost, and reason."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "crew_id": {
                            "type": "string",
                            "description": "Exact Supabase crew profile ID."
                        },
                        "name": {
                            "type": "string"
                        },
                        "role": {
                            "type": "string"
                        },
                        "hourly_rate": {
                            "type": "number"
                        },
                        "estimated_cost": {
                            "type": "number",
                            "description": "hourly_rate × event hours."
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation of why this person was chosen."
                        }
                    },
                    "required": [
                        "crew_id",
                        "name",
                        "role",
                        "hourly_rate",
                        "estimated_cost",
                        "reason"
                    ]
                }
            },
            "total_cost": {
                "type": "number",
                "description": "Sum of all estimated costs."
            },
            "budget": {
                "type": "number",
                "description": "The organizer's total budget."
            },
            "alternatives": {
                "type": "array",
                "description": "Alternative crew members for any role (e.g. Budget Friendly or Alternative Team).",
                "items": {
                    "type": "object",
                    "properties": {
                        "crew_id": {"type": "string"},
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "hourly_rate": {"type": "number"},
                        "reason": {"type": "string"},
                        "category": {"type": "string", "description": "e.g. Budget Friendly or Premium"}
                    }
                }
            },
            "backups": {
                "type": "array",
                "description": "Backup candidate options for each required role in case primary crew cancels.",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "primary_crew_id": {"type": "string"},
                        "primary_name": {"type": "string"},
                        "backup_crew_id": {"type": "string"},
                        "backup_name": {"type": "string"},
                        "backup_rate": {"type": "number"},
                        "reason": {"type": "string"}
                    }
                }
            }
        },
        "required": [
            "recommended_team",
            "total_cost",
            "budget"
        ]
    }
}


# ============================================================
# TOOL 5: CANCEL BOOKING
# ============================================================

cancel_booking_tool = {
    "type": "function",
    "name": "cancel_booking",
    "description": (
        "Cancel a specific crew member's booking in Supabase. "
        "Use this whenever an organizer or crew member indicates a booking cancellation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "booking_id": {
                "type": "string",
                "description": "Supabase booking ID to cancel."
            }
        },
        "required": ["booking_id"]
    }
}


# ============================================================
# TOOL 6: FIND REPLACEMENT CANDIDATES
# ============================================================

find_replacement_candidates_tool = {
    "type": "function",
    "name": "find_replacement_candidates",
    "description": (
        "Retrieve replacement crew candidates for a role, excluding "
        "already-booked or cancelled crew members. Use this when a crew member "
        "cancels and a replacement needs to be recommended."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "event_id": {"type": "string"},
            "event_date": {"type": "string"},
            "start_datetime": {"type": "string"},
            "end_datetime": {"type": "string"},
            "budget": {"type": "number"},
            "exclude_crew_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of crew IDs to exclude from replacement search."
            }
        },
        "required": ["role", "event_id", "event_date", "start_datetime", "end_datetime", "budget"]
    }
}

ALL_TOOLS = [
    find_team_candidates_tool,
    find_crew_candidates_tool,
    create_event_tool,
    book_crew_member_tool,
    submit_recommendation_tool,
    cancel_booking_tool,
    find_replacement_candidates_tool,
]


# ============================================================
# HELPER: SAFE INTERACTION CREATE WITH RATE LIMIT RETRY
# ============================================================

def safe_create_interaction(**kwargs):
    """
    Wrapper around client.interactions.create that handles 429 rate limit
    errors with fast-fail retry logic (max 2 retries, short waits).
    
    For free tier Gemini API keys (15 RPM), we fail fast rather than
    sleeping for 60+ seconds which would block the entire server thread.
    """
    import time
    import re
    max_retries = 2  # Reduced from 5 to fail fast on free tier
    for attempt in range(max_retries):
        try:
            return client.interactions.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()
            is_rate_limit = (
                "429" in err_lower
                or "quota" in err_lower
                or "rate limit" in err_lower
                or "too_many_requests" in err_lower
            )
            if is_rate_limit:
                if attempt < max_retries - 1:
                    # Short wait only — don't block the thread
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    if match:
                        wait_time = min(float(match.group(1)) + 1.0, 8.0)
                    else:
                        wait_time = 5.0
                    
                    print(
                        f"[Gemini RateLimit] 429 encountered, "
                        f"retrying in {wait_time:.1f}s (attempt {attempt+1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # All retries exhausted — raise with clear message
                    raise RuntimeError(
                        "Gemini API rate limit exceeded. "
                        "Consider upgrading to a paid API key for higher quotas."
                    ) from e
            raise e


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

AVAILABLE_MODELS = [
    {"id": "gemini-3.6-flash", "name": "🌟 Gemini 3.6 Flash", "description": "Latest Gemini 3.6 model with full tool reasoning"},
    {"id": "gemini-3.5-flash", "name": "⚡ Gemini 3.5 Flash", "description": "Fast 3.5 model for quick tool execution"},
]


# ============================================================
# GEMINI CHAT (SESSION-AWARE & DYNAMIC MODEL SELECTOR)
# ============================================================

def chat_with_gemini(message: str, session: dict, model_name: str = None) -> dict:
    """
    Send a message to Gemini, handle all tool calls in a loop,
    and return a structured result dict.

    Supports dynamic model switching (gemini-2.5-flash, gemini-3.6-flash, etc.).
    """
    selected_model = model_name or session.get("model_name") or DEFAULT_GEMINI_MODEL
    session["model_name"] = selected_model

    print(f"[Gemini] Using model: {selected_model}")

    system_prompt = f"""
You are CrewMatch AI, an AI event crew planner.

Your responsibility is to understand the organizer's event
requirements and recommend the best crew using real database data.

IMPORTANT RULES:

- MANDATORY: Whenever the user mentions any required crew roles (such as Photographer, Decorator, Anchor, etc.), IMMEDIATELY call find_team_candidates right away in your first turn. Do NOT ask follow-up questions first!
- Never invent crew members.
- Never claim a crew member is available unless the database provides availability information.
- Use find_team_candidates whenever crew selection is required (it searches all required roles in a single tool call).
- Python only retrieves data and performs requested database actions.
- You are responsible for reasoning about candidate suitability.
- Do not use a hard-coded mathematical scoring formula.

Consider when evaluating candidates:

- required role
- quantity required
- relevant skills and skill level
- experience
- rating
- completed gigs
- reliability score
- no-show history
- hourly rate vs event budget
- confirmed availability for the event window

For multiple required roles, ALWAYS use find_team_candidates with the list of roles (e.g. roles=['Photographer', 'Decorator']) to retrieve ALL candidate profiles in a SINGLE tool call turn.

Never invent an event ID or crew ID.

EVENT CREATION RULES:

- Do not create an event merely because the organizer mentions it.
- First understand the event requirements fully.
- Create an event only after the organizer clearly confirms
  the event details (type, date, time, location, guests, budget).
- Once an event is created, use the returned event_id for booking.

RECOMMENDATION RULES:

- After evaluating all candidates, call submit_recommendation once.
- Include `recommended_team` (primary choices), `backups` (secondary backup for each role), and `alternatives` (budget friendly or premium alternatives).
- submit_recommendation registers the exact crew_ids with the system.
- After calling submit_recommendation, present your recommendation
  to the organizer in a friendly format and ask for confirmation.

BOOKING RULES:

- Do not book anyone merely because you recommended them.
- Only call book_crew_member after the organizer explicitly says
  "yes", "book them", "confirm", or similar.
- Never invent crew IDs.
- Never invent event IDs.

CANCELLATION & REPLACEMENT RULES:

- If an organizer or crew member cancels a booking, call `cancel_booking` with the booking ID.
- Then call `find_replacement_candidates` for that role, excluding the cancelled crew ID.
- Evaluate the replacement candidates and recommend the best replacement.
- Ask the organizer for confirmation before booking the replacement.

USER MESSAGE:

{message}
"""

    # First call: new or continuing conversation.
    init_kwargs = {
        "model": selected_model,
        "input": system_prompt,
        "tools": ALL_TOOLS,
    }

    # Continue from previous turn if this session has a prior interaction.
    try:
        interaction = safe_create_interaction(**init_kwargs)
    except Exception as e:
        err_lower = str(e).lower()
        if "429" in err_lower or "quota" in err_lower or "rate limit" in err_lower or "too_many_requests" in err_lower:
            print(f"[Gemini RateLimit] Returning fallback message due to rate limit: {e}")
            return {
                "response_text": (
                    "I am currently experiencing high demand and hit the Gemini API rate limit. "
                    "Please wait a few seconds and send your request again!"
                ),
                "recommendation": None,
                "event_id": session.get("event_id"),
                "booking_triggered": False,
                "rate_limited": True
            }
        raise e

    # Accumulate all structured data produced during the tool loop.
    recommendation_data = None
    booking_triggered = False
    try:
        # Process tool calls in a loop until Gemini gives a final text answer.
        while True:

            function_calls = [
                step
                for step in interaction.steps
                if step.type == "function_call"
            ]

            if not function_calls:
                # No more tool calls — Gemini has given its final answer.
                break

            function_results = []

            for step in function_calls:

                arguments = step.arguments

                # ------------------------------------------------
                # FIND TEAM CANDIDATES (MULTI-ROLE CONSOLIDATED)
                # ------------------------------------------------

                if step.name == "find_team_candidates":

                    print(
                        f"[Gemini] Requesting consolidated candidates for roles: "
                        f"{arguments.get('roles', [])}"
                    )

                    team_data = find_team_candidates(
                        roles=arguments["roles"],
                        event_date=arguments["event_date"],
                        start_datetime=arguments["start_datetime"],
                        end_datetime=arguments["end_datetime"],
                        budget=arguments["budget"]
                    )

                    function_results.append({
                        "type": "function_result",
                        "name": step.name,
                        "call_id": step.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(team_data, default=str)
                            }
                        ]
                    })

                # ------------------------------------------------
                # FIND CREW CANDIDATES
                # ------------------------------------------------

                elif step.name == "find_crew_candidates":

                    print(
                        f"[Gemini] Requesting candidates for: "
                        f"{arguments['role']}"
                    )

                    crew_data = find_crew_candidates(
                        role=arguments["role"],
                        event_date=arguments["event_date"],
                        start_datetime=arguments["start_datetime"],
                        end_datetime=arguments["end_datetime"],
                        budget=arguments["budget"]
                    )

                    function_results.append({
                        "type": "function_result",
                        "name": step.name,
                        "call_id": step.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(crew_data, default=str)
                            }
                        ]
                    })

                # ------------------------------------------------
                # CREATE EVENT
                # ------------------------------------------------

                elif step.name == "create_event":

                    print("[Gemini] Requesting event creation.")

                    event_result = create_event(
                        event_type=arguments["event_type"],
                        event_date=arguments["event_date"],
                        start_time=arguments["start_time"],
                        end_time=arguments["end_time"],
                        location=arguments["location"],
                        guest_count=arguments["guest_count"],
                        total_budget=arguments["total_budget"]
                    )

                    # Capture the event_id into session immediately.
                    if event_result.get("success") and event_result.get("event_id"):
                        session["event_id"] = event_result["event_id"]
                        session["event_details"] = event_result.get("event")
                        print(
                            f"[Session] Event created: "
                            f"{event_result['event_id']}"
                        )

                    function_results.append({
                        "type": "function_result",
                        "name": step.name,
                        "call_id": step.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(event_result, default=str)
                            }
                        ]
                    })

                # ------------------------------------------------
                # SUBMIT RECOMMENDATION
                # ------------------------------------------------

                elif step.name == "submit_recommendation":

                    print("[Gemini] Submitting formal recommendation.")

                    team = arguments.get("recommended_team", [])
                    total_cost = arguments.get("total_cost", 0)
                    budget = arguments.get("budget", 0)
                    alternatives = arguments.get("alternatives", [])
                    backups = arguments.get("backups", [])

                    # Save to session so confirmation → booking works.
                    session["recommended_team"] = team
                    session["total_cost"] = total_cost
                    session["budget"] = budget
                    session["alternatives"] = alternatives
                    session["backups"] = backups
                    session["awaiting_confirmation"] = True

                    recommendation_data = {
                        "event_id": session.get("event_id"),
                        "recommended_team": team,
                        "total_cost": total_cost,
                        "budget": budget,
                        "budget_remaining": budget - total_cost,
                        "alternatives": alternatives,
                        "backups": backups,
                    }

                    print(
                        f"[Session] Recommendation registered: "
                        f"{len(team)} crew members, "
                        f"{len(backups)} backups, "
                        f"total INR {total_cost}"
                    )

                    function_results.append({
                        "type": "function_result",
                        "name": step.name,
                        "call_id": step.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "success": True,
                                    "message": (
                                        "Recommendation registered. "
                                        "Please present it to the organizer "
                                        "and ask for confirmation."
                                    )
                                })
                            }
                        ]
                    })

                # ------------------------------------------------
                # BOOK CREW MEMBER
                # ------------------------------------------------

                elif step.name == "book_crew_member":

                    print(
                        f"[Gemini] Booking crew: "
                        f"{arguments.get('crew_id')}"
                    )

                    booking_result = book_crew_member(
                        event_id=arguments["event_id"],
                        crew_id=arguments["crew_id"],
                        role=arguments["role"],
                        agreed_rate=arguments["agreed_rate"]
                    )

                    booking_triggered = True

                    function_results.append({
                        "type": "function_result",
                        "name": step.name,
                        "call_id": step.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    booking_result,
                                    default=str
                                )
                            }
                        ]
                    })

                # ------------------------------------------------
                # CANCEL BOOKING
                # ------------------------------------------------

                elif step.name == "cancel_booking":

                    print(
                        f"[Gemini] Cancelling booking: "
                        f"{arguments.get('booking_id')}"
                    )

                    cancellation = cancel_booking(
                        booking_id=arguments["booking_id"]
                    )

                    function_results.append({
                        "type": "function_result",
                        "name": step.name,
                        "call_id": step.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    cancellation,
                                    default=str
                                )
                            }
                        ]
                    })

                # ------------------------------------------------
                # FIND REPLACEMENT CANDIDATES
                # ------------------------------------------------

                elif step.name == "find_replacement_candidates":

                    print(
                        f"[Gemini] Finding replacement candidates for role: "
                        f"{arguments.get('role')}"
                    )

                    replacements = find_replacement_candidates(
                        role=arguments["role"],
                        event_id=arguments["event_id"],
                        event_date=arguments["event_date"],
                        start_datetime=arguments["start_datetime"],
                        end_datetime=arguments["end_datetime"],
                        budget=arguments["budget"],
                        exclude_crew_ids=arguments.get("exclude_crew_ids", [])
                    )

                    function_results.append({
                        "type": "function_result",
                        "name": step.name,
                        "call_id": step.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    replacements,
                                    default=str
                                )
                            }
                        ]
                    })

            if not function_results:
                break

            # Send all tool results back to Gemini and continue.
            interaction = safe_create_interaction(
                model=selected_model,
                previous_interaction_id=interaction.id,
                input=function_results,
                tools=ALL_TOOLS,
            )

        # Save the latest interaction ID so the next turn can continue.
        session["interaction_id"] = interaction.id

        return {
            "response_text": interaction.output_text or "",
            "recommendation": recommendation_data,
            "event_id": session.get("event_id"),
            "booking_triggered": booking_triggered,
        }

    except Exception as e:
        print(f"[Gemini Loop Exception] {e}")
        # If recommendation was captured before exception, return it!
        if recommendation_data:
            return {
                "response_text": "I evaluated the candidates and selected your team. Would you like to confirm and book them?",
                "recommendation": recommendation_data,
                "event_id": session.get("event_id"),
                "booking_triggered": False,
            }
        return {
            "response_text": "The request timed out or hit API limits while communicating with Gemini. Please try again in a moment.",
            "recommendation": None,
            "event_id": session.get("event_id"),
            "booking_triggered": False,
            "error": str(e)
        }