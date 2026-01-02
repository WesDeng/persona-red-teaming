"""
Pydantic schemas for API request/response models

These schemas define the structure of data returned by the history API endpoints.
They provide automatic validation and serialization for FastAPI.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


class PersonaHistoryItem(BaseModel):
    """
    Individual persona in history with success metrics.

    Includes computed metrics like success rate (% unsafe verdicts)
    and counts of unsafe vs safe generations.
    """
    id: int
    persona_yaml: str
    emphasis_instructions: Optional[str] = None
    mutation_type: Optional[str] = None
    risk_category: Optional[str] = None
    attack_style: Optional[str] = None
    created_at: datetime
    total_generations: int  # How many times this persona was used
    success_rate: float  # Percentage of prompts marked unsafe by guard
    unsafe_count: int
    safe_count: int

    class Config:
        from_attributes = True  # Allows conversion from SQLAlchemy models


class PromptHistoryItem(BaseModel):
    """
    Individual prompt in history with its evaluation results.

    Includes the prompt text, guard verdict, user feedback, and
    information about edits and the persona that generated it.
    """
    id: int
    prompt_type: str  # 'seed', 'adversarial', 'edited'
    prompt_text: str
    seed_prompt_text: Optional[str] = None  # For adversarial prompts
    target_response: Optional[str] = None
    guard_verdict: Optional[str] = None  # 'safe' or 'unsafe'
    guard_score: Optional[float] = None  # 0.0 to 1.0
    user_marked_unsafe: Optional[bool] = None
    edit_count: int  # How many times this was edited
    created_at: datetime
    persona_yaml: str  # Include persona for context

    class Config:
        from_attributes = True


class EditHistoryItem(BaseModel):
    """
    Single item in the edit history chain.

    Represents one version of a prompt (original or edited) with
    its guard evaluation at that point in time.
    """
    id: int
    prompt_text: str
    guard_verdict: str
    guard_score: float
    created_at: datetime
    is_original: bool  # True for first version, False for edits

    class Config:
        from_attributes = True


class SessionStats(BaseModel):
    """
    Summary statistics for a user session.

    Provides aggregate metrics across all personas, generations,
    and prompts created in this session.
    """
    session_id: str
    total_personas: int
    total_generations: int
    total_prompts: int
    total_unsafe_by_guard: int  # Number marked unsafe by automated guard
    total_unsafe_by_user: int  # Number manually marked unsafe by user
    overall_success_rate: float  # Percentage
    most_successful_persona_id: Optional[int] = None  # Persona with highest success rate
    created_at: datetime
    last_active: datetime

    class Config:
        from_attributes = True


class PersonaHistoryResponse(BaseModel):
    """Response for GET /api/history/personas"""
    personas: List[PersonaHistoryItem]
    total: int


class PromptHistoryResponse(BaseModel):
    """Response for GET /api/history/prompts"""
    prompts: List[PromptHistoryItem]
    total: int


class EditHistoryResponse(BaseModel):
    """Response for GET /api/history/prompts/{prompt_id}/edits"""
    original_prompt_id: int
    edits: List[EditHistoryItem]
    total_edits: int


class MarkUnsafeRequest(BaseModel):
    """Request for POST /api/mark-unsafe"""
    prompt_id: int
    marked: bool


class MarkUnsafeResponse(BaseModel):
    """Response for POST /api/mark-unsafe"""
    status: str
    prompt_id: int
    marked_unsafe: bool
