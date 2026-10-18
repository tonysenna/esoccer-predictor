from datetime import datetime, timezone, timedelta
from src.db.session import SessionLocal
from src.db.models import Match, MatchStatus, DurationFormat, Market, OddsSnapshot
from src.db.queries import get_or_create_player
from src.analytics.dixon_coles import DixonColesModel
from src.analytics.elo import DynamicEloEngine
from src.analytics.odds import evaluate_market_value
from src.config import get_settings

settings = get_settings()

real_fixtures = [
    {"home": "Bosko", "away": "Mani", "time": "19:55", "tournament": "GT Leagues (12 min)", "format": DurationFormat.MIN_12, "odds": (2.45, 4.50, 2.05)},
    {"home": "Cira", "away": "pikalicaaa", "time": "19:55", "tournament": "GT Leagues (12 min)", "format": DurationFormat.MIN_12, "odds": (1.90, 4.75, 2.70)},
    {"home": "Mani", "away": "Ilulle", "time": "20:07", "tournament": "GT Leagues (12 min)", "format": DurationFormat.MIN_12, "odds": (2.05, 4.33, 2.50)},
    {"home": "pikalicaaa", "away": "Bosko", "time": "20:07", "tournament": "GT Leagues (12 min)", "format": DurationFormat.MIN_12, "odds": (2.00, 5.00, 2.40)},
    {"home": "Bosko", "away": "Cira", "time": "20:19", "tournament": "GT Leagues (12 min)", "format": DurationFormat.MIN_12, "odds": (2.00, 5.75, 2.25)},
    {"home": "Ilulle", "away": "pikalicaaa", "time": "20:19", "tournament": "GT Leagues (12 min)", "format": DurationFormat.MIN_12, "odds": (2.05, 4.75, 2.37)},
]

db = SessionLocal()
now = datetime.now(timezone.utc)

print("=" * 80)
print("   ESTUDIO CUANTITATIVO DE LA IA — PARTIDOS REALES DE LA CAPTURA (GT LEAGUES)")
print("=" * 80)

model_12m = DixonColesModel(duration_format=DurationFormat.MIN_12, decay_xi=settings.TIME_DECAY_XI)
elo_engine = DynamicEloEngine(settings.BASE_ELO_RATING, k_factor=settings.ELO_K_FACTOR)

results_summary = []

for idx, fix in enumerate(real_fixtures):
    p1 = get_or_create_player(db, fix["home"], source="real_feed")
    p2 = get_or_create_player(db, fix["away"], source="real_feed")
    
    # Crear o actualizar partido en BD
    m = db.query(Match).filter_by(source_match_id=f"wa_{fix['home']}_{fix['away']}_{fix['time']}").first()
    if not m:
        m = Match(
            source="real_feed",
            source_match_id=f"wa_{fix['home']}_{fix['away']}_{fix['time']}",
            tournament=fix["tournament"],
            duration_format=fix["format"],
            status=MatchStatus.SCHEDULED,
            player_home_id=p1.id,
            player_away_id=p2.id,
            scheduled_at=now + timedelta(minutes=idx * 12),
        )
        db.add(m)
        db.flush()

        o1, ox, o2 = fix["odds"]
        for sel, val in [("home", o1), ("draw", ox), ("away", o2)]:
            db.add(OddsSnapshot(
                match_id=m.id,
                bookmaker="bet365",
                market=Market.ONE_X_TWO,
                selection=sel,
                odds_value=val,
                captured_at=now,
                is_closing_line=False,
            ))
        db.commit()

    # Predicción del modelo
    r1, _, _ = elo_engine.get_player_rating(db, p1.id, fix["format"])
    r2, _, _ = elo_engine.get_player_rating(db, p2.id, fix["format"])
    pred = model_12m.predict_match(p1.id, p2.id, elo_home=r1, elo_away=r2, h2h_sample_size=0)

    # Evaluar +EV en 1X2
    o1, ox, o2 = fix["odds"]
    vbs = evaluate_market_value(
        Market.ONE_X_TWO,
        {"home": o1, "draw": ox, "away": o2},
        {"home": pred.home_win, "draw": pred.draw, "away": pred.away_win},
        min_edge=0.03,
        kelly_fraction=0.25,
    )

    best_vb = None
    for vb in vbs:
        if vb.is_value and (best_vb is None or vb.ev_edge > best_vb.ev_edge):
            best_vb = vb

    results_summary.append({
        "time": fix["time"],
        "match": f"{fix['home']} vs {fix['away']}",
        "odds_str": f"1: @{o1} | X: @{ox} | 2: @{o2}",
        "probs": f"1: {pred.home_win*100:.1f}% | X: {pred.draw*100:.1f}% | 2: {pred.away_win*100:.1f}%",
        "xg": f"{pred.expected_goals_home:.2f} - {pred.expected_goals_away:.2f} (Total: {pred.expected_total_goals:.2f})",
        "over25": f"{pred.over_2_5*100:.1f}%",
        "btts": f"{pred.btts_yes*100:.1f}%",
        "best_vb": best_vb,
    })

for res in results_summary:
    print(f"\n[{res['time']}] {res['match']}")
    print(f"   Cuotas Bookie: {res['odds_str']}")
    print(f"   Probabilidades IA: {res['probs']}")
    print(f"   xG Proyectado: {res['xg']} | Prob Mas 2.5: {res['over25']} | Ambos Marcan: {res['btts']}")
    if res["best_vb"]:
        vb = res["best_vb"]
        print(f"   [VALUE BET +EV]: Seleccion {vb.selection} @{vb.decimal_odds:.2f} | Ventaja: {vb.ev_edge*100:+.1f}% | Kelly Stake: {vb.kelly_stake_pct*100:.1f}%")
    else:
        print("   [Decision]: Sin valor suficiente en 1X2 al precio actual.")

db.close()
