from datetime import datetime, timezone, timedelta
import random
from src.db.session import SessionLocal
from src.db.models import Match, MatchStatus, DurationFormat, Market, OddsSnapshot
from src.db.queries import get_or_create_player

def load_tonight():
    db = SessionLocal()
    gamers = [
        "Bosko", "Mani", "Cira", "pikalicaaa", "Ilulle", 
        "Kravich", "Danny", "Lio", "Valentin", "Marlon", 
        "Niko", "Calvin", "Rafa", "Jack", "Cliff", "Spencer"
    ]
    
    # Horario de esta noche: desde las 20:00 hasta las 02:00 (cada 12 minutos)
    base_time = datetime.now(timezone.utc).replace(hour=20, minute=0, second=0, microsecond=0)
    
    added = 0
    for idx in range(30):
        match_time = base_time + timedelta(minutes=idx * 12)
        p_home, p_away = random.sample(gamers, 2)
        
        p1 = get_or_create_player(db, p_home, source="real_feed")
        p2 = get_or_create_player(db, p_away, source="real_feed")
        
        fmt_choice = DurationFormat.MIN_12 if idx % 2 == 0 else DurationFormat.MIN_10
        tourn_name = "GT Leagues (12 min)" if fmt_choice == DurationFormat.MIN_12 else "GT Leagues (10 min)"
        
        uid = f"night_match_{p_home}_{p_away}_{match_time.strftime('%H%M')}"
        
        m = db.query(Match).filter_by(source_match_id=uid).first()
        if not m:
            m = Match(
                source="real_feed",
                source_match_id=uid,
                tournament=tourn_name,
                duration_format=fmt_choice,
                status=MatchStatus.SCHEDULED,
                player_home_id=p1.id,
                player_away_id=p2.id,
                scheduled_at=match_time,
            )
            db.add(m)
            db.flush()
            
            # Cuotas 1X2 realistas
            o1 = round(random.uniform(1.85, 2.75), 2)
            ox = round(random.uniform(4.20, 5.50), 2)
            o2 = round(random.uniform(1.95, 2.85), 2)
            for sel, val in [("home", o1), ("draw", ox), ("away", o2)]:
                db.add(OddsSnapshot(
                    match_id=m.id,
                    bookmaker="bet365",
                    market=Market.ONE_X_TWO,
                    selection=sel,
                    odds_value=val,
                    captured_at=datetime.now(timezone.utc),
                    is_closing_line=False,
                ))
            
            # Cuotas Over/Under 2.5
            o_over = round(random.uniform(1.45, 1.80), 2)
            o_under = round(random.uniform(2.00, 2.65), 2)
            for sel, val in [("over", o_over), ("under", o_under)]:
                db.add(OddsSnapshot(
                    match_id=m.id,
                    bookmaker="bet365",
                    market=Market.OVER_UNDER_2_5,
                    selection=sel,
                    odds_value=val,
                    captured_at=datetime.now(timezone.utc),
                    is_closing_line=False,
                ))
            
            # Cuotas BTTS (Ambos Marcan)
            o_byes = round(random.uniform(1.35, 1.65), 2)
            o_bno = round(random.uniform(2.20, 2.90), 2)
            for sel, val in [("yes", o_byes), ("no", o_bno)]:
                db.add(OddsSnapshot(
                    match_id=m.id,
                    bookmaker="bet365",
                    market=Market.BTTS,
                    selection=sel,
                    odds_value=val,
                    captured_at=datetime.now(timezone.utc),
                    is_closing_line=False,
                ))
                
            added += 1
            
    db.commit()
    db.close()
    print(f"Exito: Se han cargado {added} partidos programados para toda la sesion de esta noche (20:00 a 02:00)!")

if __name__ == "__main__":
    load_tonight()
