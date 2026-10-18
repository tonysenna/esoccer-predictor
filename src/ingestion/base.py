from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from sqlalchemy.orm import Session

from src.db.models import DurationFormat, Match, OddsSnapshot, Player


class BaseIngestionAdapter(ABC):
    """
    Interfaz abstracta para adaptadores de ingesta de datos (API-first, Scrapers como fallback).
    """

    @abstractmethod
    def fetch_upcoming_matches(self, tournament: str | None = None) -> list[dict[str, Any]]:
        """Obtiene partidos programados con cuotas previas."""
        pass

    @abstractmethod
    def fetch_past_results(
        self,
        start_date: datetime,
        end_date: datetime,
        duration_format: DurationFormat | None = None,
    ) -> list[dict[str, Any]]:
        """Obtiene resultados históricos finalizados."""
        pass

    @abstractmethod
    def sync_to_db(self, session: Session, raw_data: list[dict[str, Any]]) -> tuple[int, int]:
        """Sincroniza y persiste los datos crudos en la base de datos (retorna nuevos partidos, nuevas cuotas)."""
        pass
