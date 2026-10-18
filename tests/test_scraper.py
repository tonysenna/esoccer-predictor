from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, DurationFormat, Match, Player
from src.ingestion.stealth_scraper import IntelligentEsoccerScraper, ScrapedMatchResult, StealthSession


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_stealth_session_headers_rotation():
    session = StealthSession()
    assert "User-Agent" in session.session.headers
    assert "sec-ch-ua" in session.session.headers
    assert "Accept-Language" in session.session.headers


def test_scraper_format_and_gamer_normalization(memory_db):
    scraper = IntelligentEsoccerScraper(memory_db)

    # Test Format Detection
    assert scraper.parse_format_from_text("GT Leagues - 10 mins (5x2)") == DurationFormat.MIN_10
    assert scraper.parse_format_from_text("Esports Battle 8 min (4x2)") == DurationFormat.MIN_8
    assert scraper.parse_format_from_text("Champions League 12 mins (6x2)") == DurationFormat.MIN_12

    # Test Gamer Cleaning
    assert scraper.clean_gamer_name("Lio (PSG)") == "Lio"
    assert scraper.clean_gamer_name("[GT] Danny_99 (Real Madrid)") == "Danny_99"


def test_scraper_ingestion_and_elo_integration(memory_db):
    scraper = IntelligentEsoccerScraper(memory_db)

    result = ScrapedMatchResult(
        source_id="scraped_001",
        tournament_name="GT Leagues 10 min",
        duration_format=DurationFormat.MIN_10,
        player_home="Lio",
        player_away="Danny",
        goals_home=3,
        goals_away=1,
        match_time=datetime.now(timezone.utc),
        odds_1x2=(2.10, 4.20, 2.75),
        odds_o25=(1.50, 2.40),
    )

    success = scraper.ingest_scraped_match(result)
    assert success is True

    # Duplicate rejection
    duplicate = scraper.ingest_scraped_match(result)
    assert duplicate is False

    # Check match stored and ratings updated
    match = memory_db.query(Match).filter_by(source_match_id="scraped_001").first()
    assert match is not None
    assert match.goals_home == 3
    assert match.goals_away == 1

    # Check players created
    p1 = memory_db.query(Player).filter_by(display_name="Lio").first()
    p2 = memory_db.query(Player).filter_by(display_name="Danny").first()
    assert p1 is not None and p2 is not None
