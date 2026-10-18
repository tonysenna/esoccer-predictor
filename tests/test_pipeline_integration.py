import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.analytics.elo import DynamicEloEngine
from src.analytics.metrics import calculate_h2h_metrics, calculate_player_form
from src.analytics.odds import evaluate_market_value
from src.db.models import Base, DurationFormat, Market, Match, MatchStatus, OddsSnapshot, Player
from src.ingestion.validators import (
    validate_market_odds_completeness,
    validate_match_payload,
    validate_odds_snapshot_payload,
)


@pytest.fixture
def test_db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()
    yield session
    session.close()


def test_end_to_end_pipeline_integration(test_db_session: Session):
    # 1. Ingestion Validation
    valid_match_dict = {
        "source": "gtleagues",
        "source_match_id": "int_test_01",
        "tournament": "GT Leagues 10 min",
        "duration_format": DurationFormat.MIN_10,
        "status": MatchStatus.FINISHED,
        "player_home_id": 1,
        "player_away_id": 2,
        "goals_home": 3,
        "goals_away": 1,
        "scheduled_at": datetime.now(timezone.utc),
    }
    match_val = validate_match_payload(valid_match_dict)
    assert match_val.is_valid is True

    # 2. Persist Players and Match to DB
    p1 = Player(display_name="Kravich", source="gtleagues", source_id="gt_kravich")
    p2 = Player(display_name="Danny", source="gtleagues", source_id="gt_danny")
    test_db_session.add_all([p1, p2])
    test_db_session.commit()

    match_obj = Match(
        source="gtleagues",
        source_match_id="int_test_01",
        tournament="GT Leagues 10 min",
        duration_format=DurationFormat.MIN_10,
        status=MatchStatus.FINISHED,
        player_home_id=p1.id,
        player_away_id=p2.id,
        goals_home=3,
        goals_away=1,
        scheduled_at=datetime.now(timezone.utc),
    )
    test_db_session.add(match_obj)
    test_db_session.commit()

    # 3. Dynamic ELO Update & Tracking
    elo_engine = DynamicEloEngine(base_rating=1500.0, k_factor=32.0)
    h_home, h_away = elo_engine.update_ratings_for_match(test_db_session, match_obj, commit=True)
    assert h_home.rating > 1500.0
    assert h_away.rating < 1500.0

    # 4. Form & H2H Metrics Calculation
    form_p1 = calculate_player_form(p1.id, [match_obj], DurationFormat.MIN_10)
    assert form_p1.matches_count == 1
    assert pytest.approx(form_p1.weighted_goals_for, rel=1e-4) == 3.0
    assert pytest.approx(form_p1.weighted_goals_against, rel=1e-4) == 1.0

    h2h = calculate_h2h_metrics(p1.id, p2.id, [match_obj], DurationFormat.MIN_10, min_sample_threshold=5)
    assert h2h.sample_size == 1
    assert h2h.is_sufficient_sample is False  # 1 < 5 threshold

    # 5. Bayesian Shrinkage Blend
    lam_elo, mu_elo = elo_engine.estimate_elo_goal_intensities(h_home.rating, h_away.rating, DurationFormat.MIN_10)
    lam_mle, mu_mle = 3.0, 1.0  # Hypothesized MLE

    lam_blended, mu_blended, w_mle = elo_engine.blend_goal_intensities(
        lam_mle=lam_mle,
        mu_mle=mu_mle,
        lam_elo=lam_elo,
        mu_elo=mu_elo,
        h2h_sample_size=h2h.sample_size,
        shrinkage_m=5.0,
    )
    # With n=1 and m=5, w_mle = 1/6 (~16.7%), w_elo = 5/6 (~83.3%)
    assert pytest.approx(w_mle, rel=1e-2) == 1.0 / 6.0
    assert lam_blended > 0.0 and mu_blended > 0.0

    # 6. Odds Validation and Market Value Detection
    odds_dict = {"home": 2.20, "draw": 4.10, "away": 2.80}
    odds_val = validate_market_odds_completeness(Market.ONE_X_TWO, odds_dict)
    assert odds_val.is_valid is True

    model_probs = {"home": 0.50, "draw": 0.22, "away": 0.28}
    value_bets = evaluate_market_value(Market.ONE_X_TWO, odds_dict, model_probs, min_edge=0.03)
    assert len(value_bets) == 3
    home_bet = next(b for b in value_bets if b.selection == "home")
    assert home_bet.is_value is True  # 0.50 * 2.20 - 1 = +10% EV
