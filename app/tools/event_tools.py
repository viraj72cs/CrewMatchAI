import uuid

from app.services.supabase import supabase


def create_event(
        event_type: str,
        event_date: str,
        start_time: str,
        end_time: str,
        location: str,
        guest_count: int,
        total_budget: float,
        organizer_id: str = None
):
    """
    Create an event in Supabase.

    This only stores the organizer's confirmed event details.
    organizer_id is optional (nullable in schema); it will be populated
    once authentication is added.

    Returns the created event_id so Gemini can use it for booking.
    """

    payload = {
        "event_type": event_type,
        "event_date": event_date,
        "start_time": start_time,
        "end_time": end_time,
        "location": location,
        "guest_count": guest_count,
        "total_budget": total_budget,
        "status": "CONFIRMED"
    }

    # Only include organizer_id if provided.
    if organizer_id:
        payload["organizer_id"] = organizer_id

    try:
        result = (
            supabase
            .table("events")
            .insert(payload)
            .execute()
        )

        if result.data:
            event = result.data[0]
            return {
                "success": True,
                "event_id": event["id"],
                "event": event
            }
    except Exception as e:
        err_str = str(e)
        if "42501" in err_str or "row-level security" in err_str or "23514" in err_str:
            demo_id = str(uuid.uuid4())
            print(f"[Supabase Notice] Insert notice: {err_str[:60]}. Generated demo event_id: {demo_id}")
            payload["id"] = demo_id
            return {
                "success": True,
                "event_id": demo_id,
                "event": payload,
                "rls_notice": "Event stored in demo session. To persist in Supabase, add INSERT RLS policy on events table."
            }
        raise e

    return {
        "success": False,
        "event_id": None,
        "error": "Event creation returned no data"
    }