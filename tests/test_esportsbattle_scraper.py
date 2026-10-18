import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, DurationFormat, MatchStatus, Player, Match
from src.ingestion.esportsbattle_scraper import (
    EsportsBattleScraper,
    get_team_tier_modifier,
)


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_team_tier_modifier():
    # Real Madrid should have positive top-tier boost
    assert get_team_tier_modifier("Real Madrid") == 0.20
    assert get_team_tier_modifier("Bayern") == 0.20
    assert get_team_tier_modifier("Napoli") == 0.08
    assert get_team_tier_modifier("Sassuolo") == 0.0
    assert get_team_tier_modifier(None) == 0.0


def test_esportsbattle_scraper_init(test_db):
    scraper = EsportsBattleScraper(test_db)
    assert scraper.db == test_db
    teams = scraper.get_all_teams()
    assert isinstance(teams, list)
    assert len(teams) > 0
