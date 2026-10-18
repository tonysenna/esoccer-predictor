from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

from src.db.models import Market


@dataclass
class ValueBet:
    market: Market
    selection: str  # "home" | "away" | "draw" | "over" | "under" | "yes" | "no"
    decimal_odds: float
    raw_implied_prob: float      # 1 / decimal_odds (con margen de la casa)
    fair_implied_prob: float     # Sin overround
    model_prob: float            # Probabilidad Dixon-Coles estimada
    ev_edge: float               # (model_prob * odds) - 1
    edge_percentage: float       # ev_edge * 100
    is_value: bool               # True si ev_edge >= min_edge
    kelly_stake_pct: float       # Fracción Kelly recomendada


def remove_overround_proportional(odds: list[float]) -> list[float]:
    """
    Elimina el overround/margen de la casa distribuyendo el exceso proporcionalmente.
    sum(p_i) = 1.0
    """
    raw_probs = [1.0 / o for o in odds]
    total_margin = sum(raw_probs)
    return [p / total_margin for p in raw_probs]


def remove_overround_power(odds: list[float]) -> list[float]:
    """
    Método de potencia (Power method): encuentra el exponente k tal que sum((1/O_i)^k) = 1.0.
    Ajusta mejor las cuotas altas donde las casas suelen inflar más el margen.
    """
    raw_probs = np.array([1.0 / o for o in odds])

    def f(k: float) -> float:
        return float(np.sum(raw_probs ** k) - 1.0)

    try:
        k_opt = brentq(f, 0.5, 3.0)
        fair_probs = raw_probs ** k_opt
        return list(fair_probs / np.sum(fair_probs))
    except Exception:
        return remove_overround_proportional(odds)


def remove_overround_shin(odds: list[float]) -> list[float]:
    """
    Método de Shin (1992/1993): Modelo econométrico que asume que el margen de la casa
    se compone del costo de transacción y la protección contra apostadores con información privilegiada (z).
    """
    raw_probs = np.array([1.0 / o for o in odds])
    n = len(odds)

    def f(z: float) -> float:
        # Probabilidades ajustadas según Shin
        denom = np.sqrt(z**2 + 4.0 * (1.0 - z) * raw_probs)
        return float(np.sum((denom - z) / (2.0 * (1.0 - z))) - 1.0)

    try:
        z_opt = brentq(f, 0.0, 0.4)
        denom = np.sqrt(z_opt**2 + 4.0 * (1.0 - z_opt) * raw_probs)
        p_shin = (denom - z_opt) / (2.0 * (1.0 - z_opt))
        return list(p_shin / np.sum(p_shin))
    except Exception:
        return remove_overround_power(odds)


def calculate_kelly_stake(
    model_prob: float,
    decimal_odds: float,
    kelly_fraction: float = 0.25,
    max_stake: float = 0.05,
) -> float:
    """
    Calcula el tamaño de apuesta óptimo mediante el criterio de Kelly fraccional.
    f* = (b*p - q) / b donde b = decimal_odds - 1, q = 1 - p.
    """
    if decimal_odds <= 1.0 or model_prob <= 0.0:
        return 0.0

    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    full_kelly = (b * model_prob - q) / b

    if full_kelly <= 0.0:
        return 0.0

    stake = full_kelly * kelly_fraction
    return min(stake, max_stake)


def evaluate_market_value(
    market: Market,
    selections_odds: dict[str, float],
    model_probs: dict[str, float],
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
    max_stake: float = 0.05,
    method: str = "shin",
) -> list[ValueBet]:
    """
    Evalúa un mercado completo (ej. 1X2 o Over/Under), remueve el overround de la casa,
    compara contra las probabilidades del modelo Dixon-Coles y detecta apuestas de valor (+EV).
    """
    selections = list(selections_odds.keys())
    odds_list = [selections_odds[s] for s in selections]

    # Eliminar margen
    if method == "shin":
        fair_probs_list = remove_overround_shin(odds_list)
    elif method == "power":
        fair_probs_list = remove_overround_power(odds_list)
    else:
        fair_probs_list = remove_overround_proportional(odds_list)

    results: list[ValueBet] = []

    for sel, odds_val, fair_p in zip(selections, odds_list, fair_probs_list, strict=False):
        raw_implied = 1.0 / odds_val if odds_val > 0 else 0.0
        m_prob = model_probs.get(sel, 0.0)

        # Expected Value = (model_prob * odds) - 1
        ev = (m_prob * odds_val) - 1.0
        edge_pct = ev * 100.0
        is_val = (ev >= min_edge)

        k_stake = calculate_kelly_stake(
            model_prob=m_prob,
            decimal_odds=odds_val,
            kelly_fraction=kelly_fraction,
            max_stake=max_stake,
        ) if is_val else 0.0

        results.append(
            ValueBet(
                market=market,
                selection=sel,
                decimal_odds=odds_val,
                raw_implied_prob=raw_implied,
                fair_implied_prob=fair_p,
                model_prob=m_prob,
                ev_edge=ev,
                edge_percentage=edge_pct,
                is_value=is_val,
                kelly_stake_pct=k_stake,
            )
        )

    return results
