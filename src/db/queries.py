from datetime import datetime
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.db.models import (
    DurationFormat,
    Match,
    MatchStatus,
    OddsSnapshot,
    Player,
    RatingHistory,
)


def get_h2h_matches(
    session: Session,
    player_a_id: int,
    player_b_id: int,
    duration_format: DurationFormat | None = None,
    before_date: datetime | None = None,
    limit: int = 20,
) -> list[Match]:
    """Devuelve los enfrentamientos directos entre dos gamers, en
    cualquier orden local/visitante, ordenados del más reciente al más
    antiguo. Filtra por formato de duración si se indica (recomendado:
    nunca mezclar formatos al calcular tendencias)."""
    stmt = (
        select(Match)
        .where(
            Match.status == MatchStatus.FINISHED,
            (
                ((Match.player_home_id == player_a_id) & (Match.player_away_id == player_b_id))
                | ((Match.player_home_id == player_b_id) & (Match.player_away_id == player_a_id))
            ),
        )
        .order_by(Match.scheduled_at.desc())
    )
    if duration_format is not None:
        stmt = stmt.where(Match.duration_format == duration_format)
    if before_date is not None:
        stmt = stmt.where(Match.scheduled_at < before_date)
    if limit is not None:
        stmt = stmt.limit(limit)

    return list(session.execute(stmt).scalars().all())


def get_recent_form(
    session: Session,
    player_id: int,
    duration_format: DurationFormat,
    before_date: datetime | None = None,
    limit: int = 10,
) -> list[Match]:
    """Últimos N partidos de un gamer en un formato de duración
    concreto (para racha reciente con decaimiento temporal)."""
    stmt = (
        select(Match)
        .where(
            Match.status == MatchStatus.FINISHED,
            Match.duration_format == duration_format,
            (Match.player_home_id == player_id) | (Match.player_away_id == player_id),
        )
        .order_by(Match.scheduled_at.desc())
    )
    if before_date is not None:
        stmt = stmt.where(Match.scheduled_at < before_date)
    if limit is not None:
        stmt = stmt.limit(limit)

    return list(session.execute(stmt).scalars().all())


def get_or_create_player(
    session: Session,
    display_name: str,
    source: str = "gtleagues",
    source_id: str | None = None
) -> Player:
    """Busca o crea un gamer en la base de datos."""
    s_id = source_id or display_name.lower().replace(" ", "_")
    stmt = select(Player).where(Player.source == source, Player.source_id == s_id)
    player = session.execute(stmt).scalar_one_or_none()
    if not player:
        # Check by display name
        stmt_name = select(Player).where(Player.display_name == display_name)
        player = session.execute(stmt_name).scalar_one_or_none()

    if not player:
        player = Player(
            display_name=display_name,
            source=source,
            source_id=s_id,
        )
        session.add(player)
        session.flush()
    return player


def get_player_by_name(session: Session, display_name: str) -> Player | None:
    """Obtiene un jugador por su nombre de pantalla."""
    stmt = select(Player).where(Player.display_name.ilike(f"%{display_name}%"))
    return session.execute(stmt).scalars().first()


def get_latest_player_rating(
    session: Session,
    player_id: int,
    duration_format: DurationFormat,
    before_date: datetime | None = None,
) -> RatingHistory | None:
    """Obtiene el último rating histórico registrado para un jugador y formato antes de una fecha."""
    stmt = (
        select(RatingHistory)
        .where(
            RatingHistory.player_id == player_id,
            RatingHistory.duration_format == duration_format,
        )
        .order_by(desc(RatingHistory.as_of), desc(RatingHistory.id))
    )
    if before_date is not None:
        stmt = stmt.where(RatingHistory.as_of <= before_date)

    return session.execute(stmt).scalars().first()


def get_all_matches_for_format(
    session: Session,
    duration_format: DurationFormat,
    before_date: datetime | None = None,
    limit: int | None = None,
) -> list[Match]:
    """Obtiene todos los partidos finalizados de un formato de duración para entrenamiento/calibración."""
    stmt = (
        select(Match)
        .where(
            Match.status == MatchStatus.FINISHED,
            Match.duration_format == duration_format,
            Match.goals_home.is_not(None),
            Match.goals_away.is_not(None),
        )
        .order_by(Match.scheduled_at.asc())
    )
    if before_date is not None:
        stmt = stmt.where(Match.scheduled_at < before_date)
    if limit is not None:
        stmt = stmt.limit(limit)

    return list(session.execute(stmt).scalars().all())


def get_closing_odds_for_match(
    session: Session,
    match_id: int
) -> list[OddsSnapshot]:
    """Obtiene las cuotas de cierre o las últimas disponibles para un partido."""
    stmt = (
        select(OddsSnapshot)
        .where(OddsSnapshot.match_id == match_id)
        .order_by(desc(OddsSnapshot.is_closing_line), desc(OddsSnapshot.captured_at))
    )
    return list(session.execute(stmt).scalars().all())


def get_upcoming_matches(
    session: Session,
    duration_format: DurationFormat | None = None,
    limit: int = 300,
) -> list[Match]:
    """Obtiene los partidos programados/futuros (de hoy) para análisis y predicción."""
    stmt = (
        select(Match)
        .where(Match.status == MatchStatus.SCHEDULED)
        .order_by(Match.scheduled_at.asc())
    )
    if duration_format is not None:
        stmt = stmt.where(Match.duration_format == duration_format)
    if limit is not None:
        stmt = stmt.limit(limit)

    return list(session.execute(stmt).scalars().all())

