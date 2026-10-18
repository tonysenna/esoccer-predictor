from datetime import datetime, timedelta, timezone
import pytest
import numpy as np

from src.analytics.dixon_coles import DixonColesModel, dixon_coles_tau
from src.db.models import DurationFormat, Match, MatchStatus


def test_dixon_coles_tau():
    lam, mu, rho = 1.8, 1.5, -0.05

    tau_00 = dixon_coles_tau(0, 0, lam, mu, rho)
    tau_01 = dixon_coles_tau(0, 1, lam, mu, rho)
    tau_10 = dixon_coles_tau(1, 0, lam, mu, rho)
    tau_11 = dixon_coles_tau(1, 1, lam, mu, rho)
    tau_22 = dixon_coles_tau(2, 2, lam, mu, rho)

    # Con rho < 0, marcadores 0-0 y 1-1 tienen tau > 1, mientras que 0-1 y 1-0 tienen tau < 1
    assert tau_00 > 1.0
    assert tau_01 < 1.0
    assert tau_10 < 1.0
    assert tau_11 > 1.0
    assert tau_22 == 1.0


def test_predict_probabilities_sum_to_one():
    model = DixonColesModel(duration_format=DurationFormat.MIN_10)
    pred = model.predict_match(
        player_home_id=1,
        player_away_id=2,
        elo_home=1620.0,
        elo_away=1440.0,
        h2h_sample_size=0,
    )

    sum_1x2 = pred.home_win + pred.draw + pred.away_win
    assert pytest.approx(sum_1x2, rel=1e-3) == 1.0
    assert pytest.approx(pred.over_1_5 + pred.under_1_5, rel=1e-3) == 1.0
    assert pytest.approx(pred.over_2_5 + pred.under_2_5, rel=1e-3) == 1.0
    assert pytest.approx(pred.over_3_5 + pred.under_3_5, rel=1e-3) == 1.0
    assert pytest.approx(pred.over_4_5 + pred.under_4_5, rel=1e-3) == 1.0
    assert pytest.approx(pred.btts_yes + pred.btts_no, rel=1e-3) == 1.0

    # Gamer con mayor ELO debe tener mayor probabilidad de ganar y mayores xG
    assert pred.home_win > pred.away_win
    assert pred.expected_goals_home > pred.expected_goals_away
    assert pred.mle_weight == 0.0  # Con h2h=0, peso MLE es 0


def test_duration_format_scaling():
    model_8m = DixonColesModel(duration_format=DurationFormat.MIN_8)
    model_12m = DixonColesModel(duration_format=DurationFormat.MIN_12)

    pred_8m = model_8m.predict_match(1, 2, 1500.0, 1500.0)
    pred_12m = model_12m.predict_match(1, 2, 1500.0, 1500.0)

    # Formato 12 min (6x2) debe proyectar significativamente más goles totales y mayor Over 3.5 que 8 min (4x2)
    assert pred_12m.expected_total_goals > pred_8m.expected_total_goals
    assert pred_12m.over_3_5 > pred_8m.over_3_5
    assert pred_8m.under_2_5 > pred_12m.under_2_5


def test_bayesian_shrinkage_integration():
    # Crear partidos sintéticos para que el modelo tenga parámetros MLE
    base_time = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    matches = []
    for i in range(25):
        m = Match(
            source="gtleagues",
            source_match_id=f"syn_m_{i}",
            tournament="GT 10min",
            duration_format=DurationFormat.MIN_10,
            status=MatchStatus.FINISHED,
            player_home_id=1 if i % 2 == 0 else 2,
            player_away_id=2 if i % 2 == 0 else 1,
            goals_home=3 if i % 2 == 0 else 1,
            goals_away=1 if i % 2 == 0 else 3,
            scheduled_at=base_time + timedelta(hours=i * 2),
        )
        matches.append(m)

    model = DixonColesModel(duration_format=DurationFormat.MIN_10, shrinkage_m=5.0)
    model.fit(matches)

    # Caso n=0: 100% ELO prior
    pred_0 = model.predict_match(1, 2, elo_home=1500.0, elo_away=1500.0, h2h_sample_size=0)
    assert pred_0.mle_weight == 0.0

    # Caso n=5: 50% MLE, 50% ELO (punto de equilibrio)
    pred_5 = model.predict_match(1, 2, elo_home=1500.0, elo_away=1500.0, h2h_sample_size=5)
    assert pytest.approx(pred_5.mle_weight, rel=1e-3) == 0.50

    # Caso n=20: 80% MLE (20/25)
    pred_20 = model.predict_match(1, 2, elo_home=1500.0, elo_away=1500.0, h2h_sample_size=20)
    assert pytest.approx(pred_20.mle_weight, rel=1e-3) == 0.80


def test_gamma_likelihood_ratio_test():
    # Generar partidos sin ventaja de local/saque (goles simétricos)
    base_time = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    matches = []
    for i in range(40):
        gh = 2 if i % 2 == 0 else 1
        ga = 1 if i % 2 == 0 else 2
        m = Match(
            source="gtleagues",
            source_match_id=f"lrt_m_{i}",
            tournament="GT 10min",
            duration_format=DurationFormat.MIN_10,
            status=MatchStatus.FINISHED,
            player_home_id=(i % 4) + 1,
            player_away_id=((i + 1) % 4) + 1,
            goals_home=gh,
            goals_away=ga,
            scheduled_at=base_time + timedelta(hours=i * 2),
        )
        matches.append(m)

    model = DixonColesModel(duration_format=DurationFormat.MIN_10, auto_test_gamma=True)
    lrt_result = model.test_gamma_significance(matches)

    assert lrt_result.statistic >= 0.0
    assert 0.0 <= lrt_result.p_value <= 1.0
    assert lrt_result.decision in ["retain_gamma", "set_gamma_zero"]


def test_rolling_window_and_batch_retrain():
    base_time = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    matches = []
    for i in range(250):  # 250 partidos generados
        m = Match(
            source="gtleagues",
            source_match_id=f"rw_m_{i}",
            tournament="GT 10min",
            duration_format=DurationFormat.MIN_10,
            status=MatchStatus.FINISHED,
            player_home_id=(i % 5) + 1,
            player_away_id=((i + 2) % 5) + 1,
            goals_home=2,
            goals_away=2,
            scheduled_at=base_time + timedelta(hours=i),
        )
        matches.append(m)

    model = DixonColesModel(duration_format=DurationFormat.MIN_10, window_size=200, retrain_batch_size=10)
    # Fit inicial con 250 partidos -> la ventana recorta a los últimos 200
    model.fit(matches)
    assert model.last_fitted_match_count == 200

    # Añadir 5 partidos más (no llega al batch_size de 10) -> no reentrena
    model.fit(matches + matches[:5])
    assert model.last_fitted_match_count == 200

    # Añadir 15 partidos más -> supera batch_size -> reentrena
    model.fit(matches + matches[:15])
    assert model.last_fitted_match_count == 200
