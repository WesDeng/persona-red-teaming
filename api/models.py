"""
SQLAlchemy ORM models for history tracking

This module defines the database schema for tracking:
- User sessions (anonymous via UUID)
- Personas created and their configurations
- Generation attempts (linking personas to prompts)
- Prompts (seed, adversarial, edited) with edit history
- Target LLM responses
- Guard evaluation results
- User feedback (mark as unsafe)
- Data retention policies
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float,
    Boolean, DateTime, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Session(Base):
    """
    Represents an anonymous user session tracked by UUID.

    Session IDs are generated in the frontend and stored in localStorage.
    No authentication or PII is collected.
    """
    __tablename__ = 'sessions'

    id = Column(String, primary_key=True)  # UUID from frontend
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    session_metadata = Column(Text)  # JSON string for future extensibility (renamed to avoid SQLAlchemy conflict)

    # Relationships
    personas = relationship("Persona", back_populates="session", cascade="all, delete-orphan")
    generations = relationship("Generation", back_populates="session", cascade="all, delete-orphan")
    user_feedback = relationship("UserFeedback", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_sessions_last_active', 'last_active'),
    )

    def __repr__(self):
        return f"<Session(id={self.id}, created_at={self.created_at})>"


class Persona(Base):
    """
    Stores persona definitions used for generating adversarial prompts.

    A persona includes demographic information, behavioral traits, and
    mutation configuration (persona-based, rainbow, risk-category).
    """
    __tablename__ = 'personas'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False)
    persona_yaml = Column(Text, nullable=False)  # Full YAML persona text
    emphasis_instructions = Column(Text)  # Additional emphasis instructions
    mutation_type = Column(String)  # 'persona', 'rainbow', 'risk-category'
    risk_category = Column(String)  # For rainbow/risk-category types
    attack_style = Column(String)  # For rainbow/risk-category types
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="personas")
    generations = relationship("Generation", back_populates="persona", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_personas_session', 'session_id', 'created_at'),
    )

    def __repr__(self):
        return f"<Persona(id={self.id}, mutation_type={self.mutation_type})>"


class Generation(Base):
    """
    Represents a single /api/generate call that creates multiple prompts.

    Links a persona to the prompts it generated, tracking the generation
    parameters like seed mode and number of mutations.
    """
    __tablename__ = 'generations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False)
    persona_id = Column(Integer, ForeignKey('personas.id', ondelete='CASCADE'), nullable=False)
    seed_mode = Column(String, nullable=False)  # 'random' or 'preselected'
    num_seed_prompts = Column(Integer)
    num_mutations_per_seed = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="generations")
    persona = relationship("Persona", back_populates="generations")
    prompts = relationship("Prompt", back_populates="generation", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_generations_session', 'session_id', 'created_at'),
        Index('idx_generations_persona', 'persona_id'),
    )

    def __repr__(self):
        return f"<Generation(id={self.id}, persona_id={self.persona_id}, seed_mode={self.seed_mode})>"


class Prompt(Base):
    """
    Stores all prompts: seed prompts, adversarial prompts, and edited prompts.

    Edit history is tracked via parent_prompt_id (forming a linked list).
    Adversarial prompts link back to their seed via seed_prompt_id.
    """
    __tablename__ = 'prompts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(Integer, ForeignKey('generations.id', ondelete='CASCADE'), nullable=False)
    prompt_type = Column(String, nullable=False)  # 'seed', 'adversarial', 'edited'
    prompt_text = Column(Text, nullable=False)
    parent_prompt_id = Column(Integer, ForeignKey('prompts.id', ondelete='SET NULL'))  # For edit history
    seed_prompt_id = Column(Integer, ForeignKey('prompts.id', ondelete='CASCADE'))  # Link to seed
    mutation_index = Column(Integer)  # Which mutation number (0-based) for this seed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    generation = relationship("Generation", back_populates="prompts")
    parent = relationship("Prompt", remote_side=[id], foreign_keys=[parent_prompt_id])
    target_responses = relationship("TargetResponse", back_populates="prompt", cascade="all, delete-orphan")
    guard_results = relationship("GuardResult", back_populates="prompt", cascade="all, delete-orphan")
    user_feedback = relationship("UserFeedback", back_populates="prompt", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_prompts_generation', 'generation_id'),
        Index('idx_prompts_parent', 'parent_prompt_id'),
        Index('idx_prompts_seed', 'seed_prompt_id'),
    )

    def __repr__(self):
        return f"<Prompt(id={self.id}, type={self.prompt_type})>"


class TargetResponse(Base):
    """
    Stores responses from the target LLM to adversarial prompts.

    Each prompt may have multiple responses if edited/reattacked.
    """
    __tablename__ = 'target_responses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('prompts.id', ondelete='CASCADE'), nullable=False)
    response_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    prompt = relationship("Prompt", back_populates="target_responses")
    guard_results = relationship("GuardResult", back_populates="target_response", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_target_responses_prompt', 'prompt_id'),
    )

    def __repr__(self):
        return f"<TargetResponse(id={self.id}, prompt_id={self.prompt_id})>"


class GuardResult(Base):
    """
    Stores guard evaluation results (safe/unsafe classification).

    Links a prompt and its target response to the guard's verdict and score.
    """
    __tablename__ = 'guard_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey('prompts.id', ondelete='CASCADE'), nullable=False)
    target_response_id = Column(Integer, ForeignKey('target_responses.id', ondelete='CASCADE'), nullable=False)
    verdict = Column(String, nullable=False)  # 'safe' or 'unsafe'
    score = Column(Float, nullable=False)  # 0.0 to 1.0
    is_harmful = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    prompt = relationship("Prompt", back_populates="guard_results")
    target_response = relationship("TargetResponse", back_populates="guard_results")

    __table_args__ = (
        Index('idx_guard_results_prompt', 'prompt_id'),
        Index('idx_guard_results_verdict', 'verdict'),
    )

    def __repr__(self):
        return f"<GuardResult(id={self.id}, verdict={self.verdict}, score={self.score})>"


class UserFeedback(Base):
    """
    Stores user's manual safety assessments (mark as unsafe button).

    Allows comparison between automated guard verdicts and human judgment.
    """
    __tablename__ = 'user_feedback'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False)
    prompt_id = Column(Integer, ForeignKey('prompts.id', ondelete='CASCADE'), nullable=False)
    marked_unsafe = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="user_feedback")
    prompt = relationship("Prompt", back_populates="user_feedback")

    __table_args__ = (
        Index('idx_user_feedback_session', 'session_id'),
        Index('idx_user_feedback_prompt', 'prompt_id'),
    )

    def __repr__(self):
        return f"<UserFeedback(id={self.id}, marked_unsafe={self.marked_unsafe})>"


class RetentionPolicy(Base):
    """
    Defines data retention policies for automatic cleanup.

    Default policy: delete sessions and all related data after 90 days of inactivity.
    """
    __tablename__ = 'retention_policy'

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_name = Column(String, nullable=False)
    retention_days = Column(Integer, nullable=False)
    last_cleanup = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<RetentionPolicy(name={self.policy_name}, days={self.retention_days})>"
