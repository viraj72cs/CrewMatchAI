"""
Pydantic response schemas for the CrewMatch API.

These are used by FastAPI for documentation and by Flutter for
understanding the expected response structure.
"""

from pydantic import BaseModel
from typing import Optional


class RecommendedCrewMember(BaseModel):
    crew_id: str
    name: str
    role: str
    hourly_rate: float
    estimated_cost: float
    reason: str


class AlternativeCrewMember(BaseModel):
    crew_id: str
    name: str
    role: str
    hourly_rate: float
    reason: str
    category: Optional[str] = "Alternative"


class BackupCrewMember(BaseModel):
    role: str
    primary_crew_id: str
    primary_name: str
    backup_crew_id: str
    backup_name: str
    backup_rate: float
    reason: str


class RecommendationResult(BaseModel):
    event_id: Optional[str]
    recommended_team: list[RecommendedCrewMember]
    total_cost: float
    budget: float
    budget_remaining: float
    alternatives: list[AlternativeCrewMember] = []
    backups: list[BackupCrewMember] = []


class BookedMember(BaseModel):
    crew_id: str
    name: str
    role: str


class BookingFailure(BaseModel):
    crew_id: str
    name: str
    role: str
    error: str


class BookingResult(BaseModel):
    success: bool
    total_requested: int
    total_booked: int
    total_failed: int
    bookings: list[BookedMember]
    failures: list[BookingFailure]


class ChatResponse(BaseModel):
    success: bool
    session_id: str
    state: str  # chatting | recommending | awaiting_confirmation | booking_complete | booking_partial | error
    message: str
    recommendation: Optional[RecommendationResult] = None
    booking_result: Optional[BookingResult] = None


class CancellationRequest(BaseModel):
    booking_id: str


class ReplacementRequest(BaseModel):
    booking_id: str
    role: str
    event_id: str
    event_date: str
    start_datetime: str
    end_datetime: str
    budget: float
    exclude_crew_ids: list[str] = []
