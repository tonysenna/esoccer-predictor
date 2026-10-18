from datetime import datetime, timezone, timedelta
from src.db.session import SessionLocal
from src.db.models import Match, MatchStatus, DurationFormat, Market, OddsSnapshot
from src.db.queries import get_or_create_player

def load_bet365_match():
    db = SessionLocal()
    now = datetime.now(timezone.utc)

    p1 = get_or_create_player(db, "Dicca", source="bet365_live")
    p2 = get_or_create_player(db, "Jankulovski", source="bet365_live")

    # Match: Napoli (Dicca) vs Sassuolo (Jankulovski) - Esports Battle 8 min
    uid = f"bet365_Dicca_Jankulovski_{int(now.timestamp())}"
    m = Match(
        source="bet365_live",
        source_match_id=uid,
        tournament="Esports Battle (8 min)",
        duration_format=DurationFormat.MIN_8,
        status=MatchStatus.SCHEDULED,
        player_home_id=p1.id,
        player_away_id=p2.id,
        scheduled_at=now + timedelta(minutes=5),
    )
    db.add(m)
    db.flush()

    # Cuotas 1X2: 1: @2.05, X: @4.75, 2: @2.40
    for sel, val in [("home", 2.05), ("draw", 4.75), ("away", 2.40)]:
        db.add(OddsSnapshot(
            match_id=m.id,
            bookmaker="bet365",
            market=Market.ONE_X_TWO,
            selection=sel,
            odds_value=val,
            captured_at=now,
            is_closing_line=False,
        ))

    # Cuotas Over/Under 2.5: Over: @1.10, Under: @6.25
    for sel, val in [("over", 1.10), ("under", 6.25)]:
        db.add(OddsSnapshot(
            match_id=m.id,
            bookmaker="bet365",
            market=Market.OVER_UNDER_2_5,
            selection=sel,
            odds_value=val,
            captured_at=now,
            is_closing_line=False,
        ))

    db.commit()
    db.close()
    print("Partido Dicca vs Jankulovski cargado exitosamente en la base de datos!")

if __name__ == "__main__":
    load_bet365_match()
