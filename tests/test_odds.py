import pytest
from src.analytics.odds import (
    calculate_kelly_stake,
    evaluate_market_value,
    remove_overround_power,
    remove_overround_proportional,
    remove_overround_shin,
)
from src.db.models import Market


def test_remove_overround_methods():
    # Odds with ~8% overround (1/2.0 + 1/3.4 + 1/3.6 = 0.50 + 0.294 + 0.278 = 1.072)
    odds = [2.0, 3.4, 3.6]

    p_prop = remove_overround_proportional(odds)
    assert pytest.approx(sum(p_prop), rel=1e-4) == 1.0

    p_power = remove_overround_power(odds)
    assert pytest.approx(sum(p_power), rel=1e-4) == 1.0

    p_shin = remove_overround_shin(odds)
    assert pytest.approx(sum(p_shin), rel=1e-4) == 1.0


def test_calculate_kelly_stake():
    # 55% model prob on 2.0 odds (+10% EV)
    # b = 1, p = 0.55, q = 0.45 -> full kelly = (1 * 0.55 - 0.45) / 1 = 0.10 (10%)
    # quarter kelly = 2.5%
    stake = calculate_kelly_stake(model_prob=0.55, decimal_odds=2.0, kelly_fraction=0.25, max_stake=0.05)
    assert pytest.approx(stake, rel=1e-3) == 0.025

    # Negative EV should give 0.0 stake
    stake_neg = calculate_kelly_stake(model_prob=0.40, decimal_odds=2.0, kelly_fraction=0.25)
    assert stake_neg == 0.0


def test_evaluate_market_value():
    odds_dict = {"home": 2.20, "draw": 3.80, "away": 3.10}
    # Model predicts Home has 52% chance
    model_probs = {"home": 0.52, "draw": 0.25, "away": 0.23}

    value_bets = evaluate_market_value(
        Market.ONE_X_TWO,
        odds_dict,
        model_probs,
        min_edge=0.03,
        kelly_fraction=0.25,
    )

    home_bet = next(b for b in value_bets if b.selection == "home")
    assert home_bet.is_value is True
    # EV = 0.52 * 2.20 - 1 = 1.144 - 1 = +14.4%
    assert pytest.approx(home_bet.ev_edge, rel=1e-2) == 0.144
    assert home_bet.kelly_stake_pct > 0.0
