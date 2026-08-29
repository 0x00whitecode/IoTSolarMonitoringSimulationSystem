"""Database initialization and session management using SQLAlchemy + SQLite."""
from __future__ import annotations
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base

logger = logging.getLogger("smart_solar_iot.database")


def get_engine(db_path: str = "data/generated/solar_iot.db"):
    """Create SQLite engine and ensure directory exists."""
    p = Path(db_path)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p
    p.parent.mkdir(parents=True, exist_ok=True)
    # SQLite with WAL mode for better concurrency
    url = f"sqlite:///{p}"
    engine = create_engine(url, echo=False)
    return engine


def init_db(db_path: str = "data/generated/solar_iot.db"):
    """Create all tables. Call once at startup."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    logger.info(f"Database initialized at {db_path}")
    return engine


def get_session(engine) -> Session:
    """Create a new database session."""
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return SessionLocal()
