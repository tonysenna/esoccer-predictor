from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from src.db.models import DurationFormat, Market, MatchStatus


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]


def validate_match_payload(data: dict[str, Any]) -> ValidationResult:
    """
    Valida la integridad de un partido antes de persistirlo en la base de datos.
    Reglas:
    - Identificadores presentes y jugadores distintos.
    - Formato de duración válido en {'4x2', '5x2', '6x2'}.
    - Goles >= 0 y enteros si el estado es FINISHED.
    - Timestamp programado válido.
    """
    errors: list[str] = []

    # Validar fuente e id
    if not data.get("source") or not data.get("source_match_id"):
        errors.append("Falta 'source' o 'source_match_id'")

    # Validar jugadores
    p_home = data.get("player_home_id")
    p_away = data.get("player_away_id")
    if p_home is None or p_away is None:
        errors.append("Falta identificador de jugador local o visitante")
    elif p_home == p_away:
        errors.append(f"Jugador local y visitante no pueden ser idénticos (ID: {p_home})")

    # Validar formato
    fmt = data.get("duration_format")
    if isinstance(fmt, str):
        valid_vals = [e.value for e in DurationFormat]
        if fmt not in valid_vals:
            errors.append(f"Formato de duración desconocido: {fmt}. Válidos: {valid_vals}")
    elif not isinstance(fmt, DurationFormat):
        errors.append(f"Tipo de formato de duración inválido: {type(fmt)}")

    # Validar goles y estado
    status = data.get("status", MatchStatus.SCHEDULED)
    gh = data.get("goals_home")
    ga = data.get("goals_away")

    if status == MatchStatus.FINISHED:
        if gh is None or ga is None:
            errors.append("Un partido finalizado debe incluir 'goals_home' y 'goals_away'")
        else:
            if not isinstance(gh, int) or gh < 0:
                errors.append(f"Goles locales inválidos: {gh} (debe ser entero >= 0)")
            if not isinstance(ga, int) or ga < 0:
                errors.append(f"Goles visitantes inválidos: {ga} (debe ser entero >= 0)")
            if gh > 30 or ga > 30:
                errors.append(f"Goles fuera de rango razonable en FIFA Esports: {gh}-{ga}")

    # Validar fechas
    sched = data.get("scheduled_at")
    fin = data.get("finished_at")
    if not isinstance(sched, datetime):
        errors.append("Falta o es inválido 'scheduled_at'")
    if fin and isinstance(sched, datetime) and fin < sched:
        errors.append("La fecha de finalización no puede ser anterior a la programada")

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)


def validate_odds_snapshot_payload(data: dict[str, Any]) -> ValidationResult:
    """
    Valida un snapshot de cuotas de bookmaker.
    Reglas:
    - Cuota decimal finita en rango [1.01, 50.0].
    - Mercado y selección reconocidos.
    """
    errors: list[str] = []

    odds = data.get("odds_value")
    if odds is None:
        errors.append("Cuota no puede ser nula")
    elif not isinstance(odds, (int, float)) or odds < 1.01 or odds > 50.0:
        errors.append(f"Cuota fuera de rango razonable: {odds} (esperado entre 1.01 y 50.0)")

    market = data.get("market")
    selection = data.get("selection")

    if not market or not selection:
        errors.append("Falta 'market' o 'selection'")

    # Validar selecciones coherentes según mercado
    if market == Market.ONE_X_TWO and selection not in ["home", "draw", "away"]:
        errors.append(f"Selección 1X2 inválida: {selection}")
    elif market in [Market.OVER_UNDER_1_5, Market.OVER_UNDER_2_5, Market.OVER_UNDER_3_5, Market.OVER_UNDER_4_5] and selection not in ["over", "under"]:
        errors.append(f"Selección Over/Under inválida: {selection}")
    elif market == Market.BTTS and selection not in ["yes", "no"]:
        errors.append(f"Selección BTTS inválida: {selection}")

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)


def validate_market_odds_completeness(market: Market, odds_dict: dict[str, float]) -> ValidationResult:
    """
    Valida que un mercado tenga todas sus ramas necesarias y que el overround sea coherente.
    """
    errors: list[str] = []
    expected_selections = {
        Market.ONE_X_TWO: {"home", "draw", "away"},
        Market.OVER_UNDER_1_5: {"over", "under"},
        Market.OVER_UNDER_2_5: {"over", "under"},
        Market.OVER_UNDER_3_5: {"over", "under"},
        Market.OVER_UNDER_4_5: {"over", "under"},
        Market.BTTS: {"yes", "no"},
    }

    req = expected_selections.get(market, set())
    missing = req - set(odds_dict.keys())
    if missing:
        errors.append(f"Faltan selecciones para el mercado {market.value}: {missing}")
        return ValidationResult(is_valid=False, errors=errors)

    # Validar suma de probabilidades implícitas brutas (margen)
    raw_prob_sum = sum(1.0 / odds_dict[s] for s in req if odds_dict[s] > 0)
    if raw_prob_sum < 0.95 or raw_prob_sum > 1.30:
        errors.append(f"Overround del mercado anormal: sum(1/odds) = {raw_prob_sum:.3f} (esperado entre 0.95 y 1.30)")

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)
