"""
Database connection and session management for SQLite

This module provides:
- Database engine configuration
- Session factory for database operations
- Database initialization (create tables)
- FastAPI dependency injection for database sessions
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as DBSession
from contextlib import contextmanager

# Import the Base and all models
from api.models import Base, RetentionPolicy

# Database configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/history.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Create engine with SQLite-specific configuration
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False  # Allow multi-threading (required for FastAPI)
    },
    pool_pre_ping=True,  # Verify connections before using
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_database():
    """
    Initialize database schema on startup.

    Creates all tables defined in models.py and inserts default retention policy.
    This function is idempotent - safe to call multiple times.
    """
    # Ensure data directory exists
    db_path = Path(DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Insert default retention policy if not exists
    db = SessionLocal()
    try:
        existing_policy = db.query(RetentionPolicy).filter(
            RetentionPolicy.policy_name == 'default'
        ).first()

        if not existing_policy:
            default_policy = RetentionPolicy(
                policy_name='default',
                retention_days=90
            )
            db.add(default_policy)
            db.commit()
            print(f"✓ Created default retention policy: 90 days")

        print(f"✓ Database initialized at {DATABASE_PATH}")

    except Exception as e:
        print(f"Error initializing retention policy: {e}")
        db.rollback()
    finally:
        db.close()


@contextmanager
def get_db() -> DBSession:
    """
    Context manager for database sessions.

    Usage:
        with get_db() as db:
            # use db
            user = db.query(User).first()

    Automatically commits on success and rolls back on exception.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_dependency():
    """
    FastAPI dependency injection pattern for database sessions.

    Usage in routes:
        from fastapi import Depends
        from api.database import get_db_dependency

        @app.get("/endpoint")
        async def endpoint(db: DBSession = Depends(get_db_dependency)):
            # use db
            users = db.query(User).all()
            return users

    Automatically handles session lifecycle (open, commit/rollback, close).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
