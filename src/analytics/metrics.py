from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Sequence
import numpy as np

from src.db.models import DurationFormat, Match


@dataclass
class FormMetrics:
    player_id: int
    duration_format: DurationFormat
    matches_count: int
    goals_for_avg: float
    goals_against_avg: float
    weighted_goals_for: float
    weighted_goals_against: float
    over_1_5_rate: float
    over_2_5_rate: float
    over_3_5_rate: float
    over_4_5_rate: float
    btts_rate: float
    win_rate: float
    draw_rate: float
    loss_rate: float


@dataclass
class H2HMetrics:
    player_a_id: int
    player_b_id: int
    duration_format: DurationFormat | None
    sample_size: int
    a_wins: int
    draws: int
    b_wins: int
    a_goals_avg: float
    b_goals_avg: float
    total_goals_avg: float
    over_2_5_rate: float
    over_3_5_rate: float
    btts_rate: float
    is_sufficient_sample: bool  # True if >= 5 matches


def calculate_player_form(
    player_id: int,
    matches: Sequence[Match],
    duration_format: DurationFormat,
    decay_xi: float = 0.05,  # exponential weight decay per match position or day
    as_of: datetime | None = None,
) -> FormMetrics:
    """
    Calcula las métricas de rendimiento y forma de un jugador considerando decaimiento temporal.
    Nunca se mezclan formatos de duración.
    """
    if not matches:
        return FormMetrics(
            player_id=player_id,
            duration_format=duration_format,
            matches_count=0,
            goals_for_avg=0.0,
            goals_against_avg=0.0,
            weighted_goals_for=0.0,
            weighted_goals_against=0.0,
            over_1_5_rate=0.0,
            over_2_5_rate=0.0,
            over_3_5_rate=0.0,
            over_4_5_rate=0.0,
            btts_rate=0.0,
            win_rate=0.0,
            draw_rate=0.0,
            loss_rate=0.0,
        )

    ref_date = as_of or datetime.now(timezone.utc)
    valid_matches = [
        m for m in matches
        if m.goals_home is not None and m.goals_away is not None
        and (m.player_home_id == player_id or m.player_away_id == player_id)
        and m.duration_format == duration_format
    ]

    if not valid_matches:
        return FormMetrics(
            player_id=player_id,
            duration_format=duration_format,
            matches_count=0,
            goals_for_avg=0.0,
            goals_against_avg=0.0,
            weighted_goals_for=0.0,
            weighted_goals_against=0.0,
            over_1_5_rate=0.0,
            over_2_5_rate=0.0,
            over_3_5_rate=0.0,
            over_4_5_rate=0.0,
            btts_rate=0.0,
            win_rate=0.0,
            draw_rate=0.0,
            loss_rate=0.0,
        )

    # Ordenar por fecha cronológica descendente (más reciente primero)
    sorted_matches = sorted(
        valid_matches,
        key=lambda m: m.scheduled_at.replace(tzinfo=timezone.utc) if m.scheduled_at.tzinfo is None else m.scheduled_at,
        reverse=True,
    )

    gf_list: list[float] = []
    ga_list: list[float] = []
    weights: list[float] = []
    over_1_5_count = 0
    over_2_5_count = 0
    over_3_5_count = 0
    over_4_5_count = 0
    btts_count = 0
    wins = 0
    draws = 0
    losses = 0

    for m in sorted_matches:
        is_home = (m.player_home_id == player_id)
        gf = float(m.goals_home if is_home else m.goals_away)
        ga = float(m.goals_away if is_home else m.goals_home)
        tot_goals = gf + ga

        gf_list.append(gf)
        ga_list.append(ga)

        m_date = m.scheduled_at
        if m_date.tzinfo is None:
            m_date = m_date.replace(tzinfo=timezone.utc)
        days_diff = max(0.0, (ref_date - m_date).total_seconds() / 86400.0)

        # Decaimiento temporal exponencial puro por días reales
        weight = math.exp(-decay_xi * days_diff)
        weights.append(weight)

        if tot_goals > 1.5:
            over_1_5_count += 1
        if tot_goals > 2.5:
            over_2_5_count += 1
        if tot_goals > 3.5:
            over_3_5_count += 1
        if tot_goals > 4.5:
            over_4_5_count += 1
        if gf > 0 and ga > 0:
            btts_count += 1

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    n = len(gf_list)
    if n == 0:
        return FormMetrics(
            player_id=player_id,
            duration_format=duration_format,
            matches_count=0,
            goals_for_avg=0.0,
            goals_against_avg=0.0,
            weighted_goals_for=0.0,
            weighted_goals_against=0.0,
            over_1_5_rate=0.0,
            over_2_5_rate=0.0,
            over_3_5_rate=0.0,
            over_4_5_rate=0.0,
            btts_rate=0.0,
            win_rate=0.0,
            draw_rate=0.0,
            loss_rate=0.0,
        )

    w_arr = np.array(weights)
    sum_w = float(np.sum(w_arr))
    if sum_w > 0:
        w_gf = float(np.sum(np.array(gf_list) * w_arr) / sum_w)
        w_ga = float(np.sum(np.array(ga_list) * w_arr) / sum_w)
    else:
        w_gf = float(np.mean(gf_list))
        w_ga = float(np.mean(ga_list))

    return FormMetrics(
        player_id=player_id,
        duration_format=duration_format,
        matches_count=n,
        goals_for_avg=float(np.mean(gf_list)),
        goals_against_avg=float(np.mean(ga_list)),
        weighted_goals_for=w_gf,
        weighted_goals_against=w_ga,
        over_1_5_rate=over_1_5_count / n,
        over_2_5_rate=over_2_5_count / n,
        over_3_5_rate=over_3_5_count / n,
        over_4_5_rate=over_4_5_count / n,
        btts_rate=btts_count / n,
        win_rate=wins / n,
        draw_rate=draws / n,
        loss_rate=losses / n,
    )


def calculate_h2h_metrics(
    player_a_id: int,
    player_b_id: int,
    matches: Sequence[Match],
    duration_format: DurationFormat | None = None,
    min_sample_threshold: int = 5,
) -> H2HMetrics:
    """Calcula las métricas de enfrentamientos directos entre Gamer A y Gamer B."""
    valid_matches = [
        m for m in matches
        if m.goals_home is not None and m.goals_away is not None
        and (duration_format is None or m.duration_format == duration_format)
    ]

    n = len(valid_matches)
    if n == 0:
        return H2HMetrics(
            player_a_id=player_a_id,
            player_b_id=player_b_id,
            duration_format=duration_format,
            sample_size=0,
            a_wins=0,
            draws=0,
            b_wins=0,
            a_goals_avg=0.0,
            b_goals_avg=0.0,
            total_goals_avg=0.0,
            over_2_5_rate=0.0,
            over_3_5_rate=0.0,
            btts_rate=0.0,
            is_sufficient_sample=False,
        )

    a_wins = 0
    draws = 0
    b_wins = 0
    a_goals: list[int] = []
    b_goals: list[int] = []
    over_2_5 = 0
    over_3_5 = 0
    btts = 0

    for m in valid_matches:
        if m.player_home_id == player_a_id:
            ga = m.goals_home
            gb = m.goals_away
        else:
            ga = m.goals_away
            gb = m.goals_home

        a_goals.append(ga)
        b_goals.append(gb)
        tot = ga + gb

        if ga > gb:
            a_wins += 1
        elif ga == gb:
            draws += 1
        else:
            b_wins += 1

        if tot > 2.5:
            over_2_5 += 1
        if tot > 3.5:
            over_3_5 += 1
        if ga > 0 and gb > 0:
            btts += 1

    return H2HMetrics(
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        duration_format=duration_format,
        sample_size=n,
        a_wins=a_wins,
        draws=draws,
        b_wins=b_wins,
        a_goals_avg=float(np.mean(a_goals)),
        b_goals_avg=float(np.mean(b_goals)),
        total_goals_avg=float(np.mean(a_goals) + np.mean(b_goals)),
        over_2_5_rate=over_2_5 / n,
        over_3_5_rate=over_3_5 / n,
        btts_rate=btts / n,
        is_sufficient_sample=(n >= min_sample_threshold),
    )
