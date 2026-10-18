import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.analytics.elo import DynamicEloEngine
from src.db.models import Base, DurationFormat, Match, MatchStatus, Player


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_elo_updates_with_goal_margin(memory_db: Session):
    p1 = Player(display_name="Gamer A", source="test", source_id="p1")
    p2 = Player(display_name="Gamer B", source="test", source_id="p2")
    memory_db.add_all([p1, p2])
    memory_db.commit()

    elo_engine = DynamicEloEngine(base_rating=1500.0, k_factor=32.0)

    # Match 1: Player A beats Player B 4-0 (high goal difference)
    m1 = Match(
        source="test",
        source_match_id="m1",
        tournament="Test League",
        duration_format=DurationFormat.MIN_10,
        status=MatchStatus.FINISHED,
        player_home_id=p1.id,
        player_away_id=p2.id,
        goals_home=4,
        goals_away=0,
        scheduled_at=datetime.now(timezone.utc),
    )
    memory_db.add(m1)
    memory_db.flush()

    h1, h2 = elo_engine.update_ratings_for_match(memory_db, m1, commit=True)

    assert h1.rating > 1500.0
    assert h2.rating < 1500.0
    assert h1.matches_played == 1
    assert h1.rating_deviation < 350.0

    # Match 2: Player A beats Player B by 1 goal (2-1)
    m2 = Match(
        source="test",
        source_match_id="m2",
        tournament="Test League",
        duration_format=DurationFormat.MIN_10,
        status=MatchStatus.FINISHED,
        player_home_id=p1.id,
        player_away_id=p2.id,
        goals_home=2,
        goals_away=1,
        scheduled_at=datetime.now(timezone.utc),
    )
    memory_db.add(m2)
    memory_db.flush()

    h3, h4 = elo_engine.update_ratings_for_match(memory_db, m2, commit=True)
    # The gain from 4-0 should be higher than from 2-1
    gain_m1 = h1.rating - 1500.0
    gain_m2 = h3.rating - h1.rating
    assert gain_m1 > gain_m2


def test_bayesian_shrinkage_blending():
    lam_mle = 3.20
    mu_mle = 1.80
    lam_elo = 2.10
    mu_elo = 2.10

    # Caso n = 0 (Sin enfrentamientos H2H): 100% peso a ELO
    lam_b0, mu_b0, w0 = DynamicEloEngine.blend_goal_intensities(lam_mle, mu_mle, lam_elo, mu_elo, h2h_sample_size=0, shrinkage_m=5.0)
    assert w0 == 0.0
    assert pytest.approx(lam_b0, rel=1e-3) == lam_elo
    assert pytest.approx(mu_b0, rel=1e-3) == mu_elo

    # Caso n = 5 (Punto de equilibrio): 50% MLE, 50% ELO
    lam_b5, mu_b5, w5 = DynamicEloEngine.blend_goal_intensities(lam_mle, mu_mle, lam_elo, mu_elo, h2h_sample_size=5, shrinkage_m=5.0)
    assert pytest.approx(w5, rel=1e-3) == 0.50
    assert pytest.approx(lam_b5, rel=1e-3) == (3.20 + 2.10) / 2.0
    assert pytest.approx(mu_b5, rel=1e-3) == (1.80 + 2.10) / 2.0

    # Caso n = 20 (Muestra amplia): 80% MLE (20/25), 20% ELO (5/25)
    lam_b20, mu_b20, w20 = DynamicEloEngine.blend_goal_intensities(lam_mle, mu_mle, lam_elo, mu_elo, h2h_sample_size=20, shrinkage_m=5.0)
    assert pytest.approx(w20, rel=1e-3) == 0.80
    assert pytest.approx(lam_b20, rel=1e-3) == 0.80 * 3.20 + 0.20 * 2.10


def test_glicko_inactivity_rd_expansion(memory_db: Session):
    p = Player(display_name="Inactivo", source="test", source_id="p_inac")
    p_opp = Player(display_name="Rival", source="test", source_id="p_opp")
    memory_db.add_all([p, p_opp])
    memory_db.commit()

    elo_engine = DynamicEloEngine(base_rating=1500.0, base_rd=350.0, inactivity_c=8.0)
    t0 = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    # Partido en t0
    m1 = Match(
        source="test",
        source_match_id="m_inac_1",
        tournament="Test",
        duration_format=DurationFormat.MIN_10,
        status=MatchStatus.FINISHED,
        player_home_id=p.id,
        player_away_id=p_opp.id,
        goals_home=2,
        goals_away=1,
        scheduled_at=t0,
    )
    memory_db.add(m1)
    memory_db.flush()

    h_p, _ = elo_engine.update_ratings_for_match(memory_db, m1, commit=True)
    rd_after_match = h_p.rating_deviation
    assert rd_after_match < 350.0

    # Consulta 10 días después de inactividad pasando fecha explícita
    t1 = t0 + timedelta(days=10)
    r_t1, rd_t1, n_t1 = elo_engine.get_player_rating(memory_db, p.id, DurationFormat.MIN_10, as_of=t1)
    assert rd_t1 > rd_after_match
    assert rd_t1 <= 350.0

    # Consulta sin pasar as_of (default a datetime.now): debe aplicar inflación automática
    r_now, rd_now, n_now = elo_engine.get_player_rating(memory_db, p.id, DurationFormat.MIN_10, as_of=None)
    assert rd_now > rd_after_match
    assert rd_now <= 350.0
