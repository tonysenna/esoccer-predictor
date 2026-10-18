"""
Esquema de base de datos (SQLAlchemy 2.0) para el sistema de análisis
predictivo de e-Soccer/FIFA (GT Leagues, Esports Battle).

Diseño:
- `Player`: gamers reales (no clubes virtuales).
- `Match`: cada partido jugado, segmentado por formato de duración
  (CRÍTICO — nunca mezclar 4x2/5x2/6x2 en el mismo cálculo).
- `OddsSnapshot`: cuotas capturadas en distintos momentos (permite
  backtesting y detectar movimiento de línea, no solo la cuota final).
- `RatingHistory`: rating dinámico tipo ELO/Glicko por gamer y por
  formato de duración (un gamer puede rendir distinto en 8 min vs 12 min).
- El H2H NO se guarda como tabla propia: se calcula por query sobre
  `Match` (ver `get_h2h_matches`).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DurationFormat(str, enum.Enum):
    """Formato de duración del partido. Nunca mezclar entre sí en el
    modelo — el ritmo de goles cambia radicalmente entre formatos."""
    MIN_8 = "4x2"    # 4 periodos de 2 min = 8 min (o 2 tiempos de 4 min)
    MIN_10 = "5x2"   # 5 periodos de 2 min = 10 min (o 2 tiempos de 5 min)
    MIN_12 = "6x2"   # 6 periodos de 2 min = 12 min (o 2 tiempos de 6 min)


class MatchStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class Market(str, enum.Enum):
    ONE_X_TWO = "1x2"
    OVER_UNDER_1_5 = "over_under_1_5"
    OVER_UNDER_2_5 = "over_under_2_5"
    OVER_UNDER_3_5 = "over_under_3_5"
    OVER_UNDER_4_5 = "over_under_4_5"
    OVER_UNDER_5_5 = "over_under_5_5"
    BTTS = "btts"  # ambos marcan


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

class Player(Base):
    """Gamer real. La clave de negocio es (display_name, source_id, source)."""
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    display_name: Mapped[str] = mapped_column(String(120), index=True)
    source_id: Mapped[str] = mapped_column(String(120))  # id en la fuente (GTLeagues, etc.)
    source: Mapped[str] = mapped_column(String(60))       # "gtleagues" | "esportsbattle" | "betsapi"
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # Estadísticas históricas de carrera (ej: Esports Battle)
    total_matches: Mapped[int | None] = mapped_column(nullable=True, default=0)
    total_win: Mapped[int | None] = mapped_column(nullable=True, default=0)
    total_lose: Mapped[int | None] = mapped_column(nullable=True, default=0)
    total_draw: Mapped[int | None] = mapped_column(nullable=True, default=0)
    win_rate: Mapped[float | None] = mapped_column(nullable=True, default=0.0)

    matches_as_home: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.player_home_id", back_populates="player_home"
    )
    matches_as_away: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.player_away_id", back_populates="player_away"
    )
    ratings: Mapped[list["RatingHistory"]] = relationship(back_populates="player")

    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_player_source"),
    )

    def __repr__(self) -> str:
        return f"<Player id={self.id} name='{self.display_name}'>"


class Match(Base):
    """Un partido jugado (o programado). El resultado sólo se rellena
    cuando status == FINISHED."""
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(60))
    source_match_id: Mapped[str] = mapped_column(String(120))

    tournament: Mapped[str] = mapped_column(String(120))
    duration_format: Mapped[DurationFormat] = mapped_column(Enum(DurationFormat), index=True)
    status: Mapped[MatchStatus] = mapped_column(Enum(MatchStatus), default=MatchStatus.SCHEDULED)

    player_home_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    player_away_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)

    team_home: Mapped[str | None] = mapped_column(String(120), nullable=True)
    team_away: Mapped[str | None] = mapped_column(String(120), nullable=True)

    goals_home: Mapped[int | None] = mapped_column(nullable=True)
    goals_away: Mapped[int | None] = mapped_column(nullable=True)

    scheduled_at: Mapped[datetime] = mapped_column(index=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)

    player_home: Mapped["Player"] = relationship(
        "Player", foreign_keys=[player_home_id], back_populates="matches_as_home"
    )
    player_away: Mapped["Player"] = relationship(
        "Player", foreign_keys=[player_away_id], back_populates="matches_as_away"
    )
    odds_snapshots: Mapped[list["OddsSnapshot"]] = relationship(back_populates="match")

    __table_args__ = (
        UniqueConstraint("source", "source_match_id", name="uq_match_source"),
        Index("ix_match_players_format", "player_home_id", "player_away_id", "duration_format"),
    )

    def __repr__(self) -> str:
        res = f"{self.goals_home}-{self.goals_away}" if self.goals_home is not None else "vs"
        return f"<Match {self.id}: {self.player_home_id} {res} {self.player_away_id} [{self.duration_format.value}]>"


class OddsSnapshot(Base):
    """Cuota capturada en un momento dado. Guardar varias capturas por
    partido (no sólo la última) permite backtesting realista y detectar
    movimiento de línea."""
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)

    bookmaker: Mapped[str] = mapped_column(String(60), default="bet365")
    market: Mapped[Market] = mapped_column(Enum(Market), index=True)
    selection: Mapped[str] = mapped_column(String(60))  # "home" | "away" | "draw" | "over" | "under" | "yes" | "no"
    odds_value: Mapped[float] = mapped_column(Float)

    captured_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc), index=True
    )
    is_closing_line: Mapped[bool] = mapped_column(default=False)

    match: Mapped["Match"] = relationship(back_populates="odds_snapshots")

    def __repr__(self) -> str:
        return f"<Odds match={self.match_id} {self.market.value}:{self.selection}={self.odds_value:.2f}>"


class RatingHistory(Base):
    """Rating dinámico (ELO/Glicko ajustado a goles) por gamer y por
    formato de duración. Sirve de respaldo cuando el H2H directo tiene
    muestra insuficiente (<5 enfrentamientos)."""
    __tablename__ = "rating_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    duration_format: Mapped[DurationFormat] = mapped_column(Enum(DurationFormat), index=True)

    rating: Mapped[float] = mapped_column(Float)
    rating_deviation: Mapped[float] = mapped_column(Float, default=350.0)  # estilo Glicko
    matches_played: Mapped[int] = mapped_column(default=0)
    as_of: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc), index=True
    )

    player: Mapped["Player"] = relationship(back_populates="ratings")

    __table_args__ = (
        Index("ix_rating_player_format_date", "player_id", "duration_format", "as_of"),
    )

    def __repr__(self) -> str:
        return f"<Rating player={self.player_id} fmt={self.duration_format.value} r={self.rating:.1f}>"
