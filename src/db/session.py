from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.db.models import Base

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(db_url: str | None = None):
    """Crea todas las tablas si no existen."""
    target_engine = create_engine(db_url) if db_url else engine
    Base.metadata.create_all(bind=target_engine)
    return target_engine


def get_db() -> Generator[Session, None, None]:
    """Generador de sesión para dependencias o CLI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
