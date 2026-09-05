from app.services.supabase import supabase


def find_crew_candidates(
        role: str,
        event_date: str,
        start_datetime: str,
        end_datetime: str,
        budget: float
):
    """
    Retrieve real crew candidates from the CrewMatch database.

    Use this when the organizer needs crew for an event.

    Returns crew profiles, skills, and availability.
    This function only retrieves database information.
    It does not rank, score, select, or book crew.
    """

    crew_response = (
        supabase
        .table("crew_profiles")
        .select(
            "id, name, primary_role, experience_years, "
            "rating, completed_gigs, no_show_count, "
            "hourly_rate, service_radius, "
            "latitude, longitude, reliability_score"
        )
        .eq("primary_role", role)
        .execute()
    )

    candidates = crew_response.data

    for candidate in candidates:

        crew_id = candidate["id"]

        skills_response = (
            supabase
            .table("crew_skills")
            .select("skill, skill_level")
            .eq("crew_id", crew_id)
            .execute()
        )

        availability_response = (
            supabase
            .table("crew_availability")
            .select(
                "start_datetime, end_datetime, status"
            )
            .eq("crew_id", crew_id)
            .execute()
        )

        candidate["skills"] = skills_response.data
        candidate["availability"] = availability_response.data

    return {
        "role": role,
        "event_date": event_date,
        "event_start": start_datetime,
        "event_end": end_datetime,
        "budget": budget,
        "candidates": candidates
    }


def find_team_candidates(
        roles: list,
        event_date: str,
        start_datetime: str,
        end_datetime: str,
        budget: float
):
    """
    Retrieve real crew candidates for MULTIPLE roles simultaneously in a single tool call.

    Use this when the organizer needs crew for an event requiring one or more roles.

    Returns crew profiles, skills, and availability grouped by role.
    """
    results = {}
    for role in roles:
        res = find_crew_candidates(
            role=role,
            event_date=event_date,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            budget=budget
        )
        results[role] = res.get("candidates", [])

    return {
        "event_date": event_date,
        "event_start": start_datetime,
        "event_end": end_datetime,
        "total_budget": budget,
        "candidates_by_role": results
    }