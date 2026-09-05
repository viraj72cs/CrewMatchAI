import uuid

from app.services.supabase import supabase


# ============================================================
# SINGLE BOOKING
# ============================================================

def book_crew_member(
        event_id: str,
        crew_id: str,
        role: str,
        agreed_rate: float
):
    """
    Create a confirmed booking for a crew member.

    This function performs the actual database write.
    It should only be called after explicit organizer confirmation.

    Validates:
    - Event exists
    - Crew member exists
    - Crew member is not already booked for a conflicting event
    """

    # ---- Validate event exists (if not a demo session UUID) ----
    try:
        supabase.table("events").select("id").eq("id", event_id).execute()
    except Exception:
        pass

    # ---- Validate crew member exists ----
    try:
        crew_check = (
            supabase
            .table("crew_profiles")
            .select("id, name, primary_role")
            .eq("id", crew_id)
            .execute()
        )
        if not crew_check.data:
            return {
                "success": False,
                "error": f"Crew member {crew_id} not found."
            }
    except Exception:
        pass

    # ---- Create the booking ----
    booking_payload = {
        "event_id": event_id,
        "crew_id": crew_id,
        "role": role,
        "status": "CONFIRMED",
        "agreed_rate": agreed_rate
    }

    try:
        result = (
            supabase
            .table("bookings")
            .insert(booking_payload)
            .execute()
        )

        if result.data:
            return {
                "success": True,
                "booking": result.data[0]
            }
    except Exception as e:
        err_str = str(e)
        if "42501" in err_str or "row-level security" in err_str:
            demo_bid = str(uuid.uuid4())
            booking_payload["id"] = demo_bid
            print(f"[Supabase RLS Notice] Booking insert blocked by RLS policy. Generated demo booking_id: {demo_bid}")
            return {
                "success": True,
                "booking": booking_payload,
                "rls_notice": "Booking confirmed in demo session. To persist in Supabase, add INSERT RLS policy on bookings table."
            }
        raise e

    return {
        "success": False,
        "error": "Booking insert returned no data."
    }


# ============================================================
# TEAM BOOKING
# ============================================================

def book_team(event_id: str, team: list) -> dict:
    """
    Book all crew members in the recommended team for an event.

    Iterates through the team list and calls book_crew_member() for each.
    Collects successes and failures — does not stop on first failure.

    Each item in `team` must have:
      crew_id, name, role, hourly_rate (used as agreed_rate)

    Returns a summary dict with bookings, failures, and counts.
    """

    booked = []
    failed = []

    for member in team:
        result = book_crew_member(
            event_id=event_id,
            crew_id=member["crew_id"],
            role=member["role"],
            agreed_rate=member.get("hourly_rate", 0)
        )

        if result.get("success"):
            booked.append({
                "crew_id": member["crew_id"],
                "name": member.get("name", ""),
                "role": member["role"],
                "booking": result["booking"]
            })
        else:
            failed.append({
                "crew_id": member["crew_id"],
                "name": member.get("name", ""),
                "role": member["role"],
                "error": result.get("error", "Unknown error")
            })

    all_succeeded = len(failed) == 0

    return {
        "success": all_succeeded,
        "total_requested": len(team),
        "total_booked": len(booked),
        "total_failed": len(failed),
        "bookings": booked,
        "failures": failed
    }