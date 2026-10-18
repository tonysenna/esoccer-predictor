from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class BacktestBetLog:
    match_id: int
    scheduled_at: str
    duration_format: str
    home_player: str
    away_player: str
    market: str
    selection: str
    odds: float
    model_prob: float
    fair_implied_prob: float
    ev_edge: float
    stake: float
    actual_result: str
    is_won: bool
    pnl: float
    bankroll_after: float


@dataclass
class MarketPerformance:
    market: str
    total_bets: int
    won_bets: int
    win_rate: float
    total_staked: float
    total_profit: float
    roi: float
    avg_odds: float
    avg_edge: float
    brier_score: float


@dataclass
class FormatPerformance:
    duration_format: str
    total_bets: int
    won_bets: int
    win_rate: float
    total_staked: float
    total_profit: float
    roi: float
    avg_odds: float
    avg_edge: float


@dataclass
class BacktestReport:
    total_matches_evaluated: int
    total_bets_placed: int
    won_bets: int
    win_rate: float
    total_staked: float
    total_profit: float
    roi: float
    final_bankroll: float
    max_drawdown_pct: float
    sharpe_ratio: float
    global_brier_score_1x2: float
    global_brier_score_totals: float
    log_loss_1x2: float
    market_breakdown: dict[str, MarketPerformance] = field(default_factory=dict)
    format_breakdown: dict[str, FormatPerformance] = field(default_factory=dict)
    bet_history: list[BacktestBetLog] = field(default_factory=list)
    pnl_series: list[float] = field(default_factory=list)


def calculate_brier_score_binary(probs: list[float], outcomes: list[int]) -> float:
    """Calcula el Brier Score para eventos binarios (Over/Under, BTTS). Rango [0, 1], menor es mejor."""
    if not probs or len(probs) != len(outcomes):
        return 0.0
    p = np.array(probs)
    y = np.array(outcomes)
    return float(np.mean((p - y) ** 2))


def calculate_brier_score_multiclass(prob_matrix: np.ndarray, outcome_indices: list[int]) -> float:
    """
    Calcula el Brier Score multi-clase para 1X2.
    prob_matrix: array (N, 3) con [p_home, p_draw, p_away]
    outcome_indices: lista de índices [0, 1, 2]
    """
    if len(outcome_indices) == 0:
        return 0.0
    n = len(outcome_indices)
    one_hot = np.zeros((n, 3))
    for i, idx in enumerate(outcome_indices):
        one_hot[i, idx] = 1.0

    return float(np.mean(np.sum((prob_matrix - one_hot) ** 2, axis=1)))


def calculate_max_drawdown(bankroll_history: list[float]) -> float:
    """Calcula el Máximo Drawdown porcentual en la curva de capital."""
    if len(bankroll_history) < 2:
        return 0.0
    peak = bankroll_history[0]
    max_dd = 0.0
    for val in bankroll_history:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def calculate_sharpe_ratio(pnl_list: list[float], risk_free_rate: float = 0.0) -> float:
    """Calcula el ratio de Sharpe sobre retornos por apuesta."""
    if len(pnl_list) < 5:
        return 0.0
    arr = np.array(pnl_list)
    std = float(np.std(arr))
    if std == 0:
        return 0.0
    return float((np.mean(arr) - risk_free_rate) / std * np.sqrt(len(arr)))
