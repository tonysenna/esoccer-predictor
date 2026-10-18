import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
import numpy as np

from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import (
    calculate_brier_score_binary,
    calculate_brier_score_multiclass,
    calculate_max_drawdown,
)
from src.db.models import Base
from src.ingestion.mock_generator import seed_realistic_esoccer_dataset


def test_brier_scores():
    # Perfect predictions
    perfect_binary = calculate_brier_score_binary([1.0, 0.0, 1.0], [1, 0, 1])
    assert perfect_binary == 0.0

    # Worst predictions
    worst_binary = calculate_brier_score_binary([0.0, 1.0], [1, 0])
    assert worst_binary == 1.0

    # Multi-class
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    actuals = [0, 1]
    assert calculate_brier_score_multiclass(probs, actuals) == 0.0


def test_max_drawdown():
    bankrolls = [1000.0, 1200.0, 900.0, 1100.0, 800.0, 1300.0]
    # Peak is 1200, lowest after peak is 800 -> drop = 400 / 1200 = 33.33%
    dd = calculate_max_drawdown(bankrolls)
    assert pytest.approx(dd, rel=1e-2) == 33.33


def test_walk_forward_backtest_simulation_robust_assertions():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Seed 50 matches per format for a solid sample
    seed_realistic_esoccer_dataset(session, num_matches_per_format=50)

    bt_engine = BacktestEngine(
        session=session,
        initial_bankroll=1000.0,
        min_edge_threshold=0.03,
        stake_mode="flat",
        flat_stake_amount=10.0,
    )

    report = bt_engine.run(min_training_matches=10)

    # 1. Verificación de ejecución y conteos
    assert report.total_matches_evaluated == 150
    assert report.total_bets_placed > 0
    assert report.won_bets <= report.total_bets_placed

    # 2. Reconciliación matemática exacta
    sum_mkt_profit = sum(p.total_profit for p in report.market_breakdown.values())
    sum_mkt_staked = sum(p.total_staked for p in report.market_breakdown.values())
    assert pytest.approx(sum_mkt_profit, abs=1e-2) == report.total_profit
    assert pytest.approx(sum_mkt_staked, abs=1e-2) == report.total_staked

    # 3. Reconciliación de Banca Final y ROI
    assert pytest.approx(report.final_bankroll, abs=1e-2) == 1000.0 + report.total_profit
    if report.total_staked > 0:
        assert pytest.approx(report.roi, abs=1e-2) == (report.total_profit / report.total_staked) * 100.0

    # 4. Calidad predictiva: Brier Scores acotados estrictamente mejor que el azar
    # Azar binario = 0.25 (un modelo entrenado debe calibrar < 0.25)
    assert report.global_brier_score_totals < 0.25
    # Azar 3-clases 1X2 = 0.667 (un modelo calibrado con datos sintéticos debe ser < 0.70)
    assert report.global_brier_score_1x2 < 0.70

    session.close()
