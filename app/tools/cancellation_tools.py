"""
Cancellation and replacement tools for CrewMatch.

These handle the case where a booked crew member becomes unavailable
and the organizer needs a replacement.
"""

from app.services.supabase import supabase
from app.tools.planner_tools import find_crew_candidates


def cancel_booking(booking_id: str) -> dict:
    """
    Cancel a specific booking by setting its status to CANCELLED.

    Returns the updated booking record.
    """

    # Fetch the booking first to get crew/event details.
    booking_check = (
        supabase
        .table("bookings")
        .select("id, event_id, crew_id, role, status")
        .eq("id", booking_id)
        .execute()
    )

    if not booking_check.data:
        return {
            "success": False,
            "error": f"Booking {booking_id} not found."
        }

    booking = booking_check.data[0]

    if booking["status"] == "CANCELLED":
        return {
            "success": False,
            "error": f"Booking {booking_id} is already cancelled."
        }

    # Update status to CANCELLED.
    result = (
        supabase
        .table("bookings")
        .update({"status": "CANCELLED"})
        .eq("id", booking_id)
        .execute()
    )

    try:
        supabase.table("crew_availability").update({"status": "AVAILABLE"}).eq("crew_id", booking["crew_id"]).execute()
    except Exception:
        pass

    return {
        "success": True,
        "cancelled_booking": result.data[0] if result.data else booking,
        "event_id": booking["event_id"],
        "crew_id": booking["crew_id"],
        "role": booking["role"]
    }


def find_replacement_candidates(
        role: str,
        event_id: str,
        event_date: str,
        start_datetime: str,
        end_datetime: str,
        budget: float,
        exclude_crew_ids: list = None
) -> dict:
    """
    Find replacement candidates for a role, excluding already-booked
    or cancelled crew members.

    Returns candidate data for Gemini to reason over.
    """

    if exclude_crew_ids is None:
        exclude_crew_ids = []

    # Get all crew already booked for this event (any status).
    existing_bookings = (
        supabase
        .table("bookings")
        .select("crew_id")
        .eq("event_id", event_id)
        .execute()
    )

    already_booked_ids = [
        b["crew_id"]
        for b in (existing_bookings.data or [])
    ]

    # Combine with any explicitly excluded IDs.
    all_excluded = list(set(already_booked_ids + exclude_crew_ids))

    # Get all candidates for the role.
    candidates_result = find_crew_candidates(
        role=role,
        event_date=event_date,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        budget=budget
    )

    # Filter out excluded crew.
    candidates = candidates_result.get("candidates", [])
    filtered = [
        c for c in candidates
        if c["id"] not in all_excluded
    ]

    return {
        "role": role,
        "event_id": event_id,
        "event_date": event_date,
        "event_start": start_datetime,
        "event_end": end_datetime,
        "budget": budget,
        "excluded_crew_ids": all_excluded,
        "replacement_candidates": filtered
    }
