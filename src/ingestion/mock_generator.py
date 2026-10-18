from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random
from sqlalchemy.orm import Session

from src.db.models import DurationFormat, Market, Match, MatchStatus, OddsSnapshot, Player
from src.db.queries import get_or_create_player


def seed_realistic_esoccer_dataset(
    session: Session,
    num_matches_per_format: int = 120,
    source: str = "gtleagues",
) -> int:
    """
    Genera un dataset sintético realista de e-Soccer (FIFA Esports) para inicializar
    la base de datos y permitir backtesting y predicción inmediata.
    """
    # Nombres típicos de gamers de GT Leagues / Esports Battle
    gamer_names = [
        "Lio", "Kravich", "Danny", "Niko", "Calvin", "Valentin", "Marlon", "Jack",
        "Rafa", "Leo", "Spencer", "Cliff", "Aladdin", "Moose", "Panic", "Zack"
    ]

    # Crear jugadores
    players: list[Player] = []
    for name in gamer_names:
        p = get_or_create_player(session, display_name=name, source=source, source_id=f"gt_{name.lower()}")
        players.append(p)
    session.commit()

    # Tasas medias de gol por formato
    format_avg_goals = {
        DurationFormat.MIN_8: 1.75,   # ~3.5 total
        DurationFormat.MIN_10: 2.35,  # ~4.7 total
        DurationFormat.MIN_12: 3.10,  # ~6.2 total
    }

    # Habilidades latentes fijas de los gamers para generar patrones consistentes
    gamer_skill = {p.id: random.uniform(0.75, 1.35) for p in players}

    base_time = datetime.now(timezone.utc) - timedelta(days=40)
    created_matches_count = 0

    for fmt in [DurationFormat.MIN_8, DurationFormat.MIN_10, DurationFormat.MIN_12]:
        base_lambda = format_avg_goals[fmt]

        for i in range(num_matches_per_format):
            # Seleccionar 2 gamers distintos
            p_home, p_away = random.sample(players, 2)
            match_time = base_time + timedelta(hours=i * 2 + random.randint(5, 45))

            # Calcular goles con distribución Poisson realista
            skill_h = gamer_skill[p_home.id]
            skill_a = gamer_skill[p_away.id]

            lam_h = max(0.2, base_lambda * (skill_h / skill_a) * random.uniform(0.9, 1.1))
            lam_a = max(0.2, base_lambda * (skill_a / skill_h) * random.uniform(0.9, 1.1))

            goals_h = int(np_poisson(lam_h))
            goals_a = int(np_poisson(lam_a))

            import uuid
            uid = uuid.uuid4().hex[:8]
            match = Match(
                source=source,
                source_match_id=f"m_{fmt.value}_{i}_{uid}",
                tournament=f"GT Leagues {fmt.value}",
                duration_format=fmt,
                status=MatchStatus.FINISHED,
                player_home_id=p_home.id,
                player_away_id=p_away.id,
                goals_home=goals_h,
                goals_away=goals_a,
                scheduled_at=match_time,
                finished_at=match_time + timedelta(minutes=15),
            )
            session.add(match)
            session.flush()
            created_matches_count += 1

            # Generar cuotas de mercado realistas con margen de la casa (~6-8%)
            _generate_mock_odds(session, match, lam_h, lam_a)

    session.commit()
    return created_matches_count


def np_poisson(lam: float) -> int:
    """Generador Poisson simple sin dependencias externas obligatorias."""
    import math
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return max(0, k - 1)


def _generate_mock_odds(session: Session, match: Match, lam_h: float, lam_a: float):
    """Crea snapshots de cuotas para 1X2, Over/Under y BTTS."""
    # Probabilidad base aproximada de victoria
    tot_lam = lam_h + lam_a
    p_h_true = lam_h / (tot_lam + 0.5)
    p_a_true = lam_a / (tot_lam + 0.5)
    p_d_true = max(0.12, 1.0 - (p_h_true + p_a_true))

    # Normalizar true probs
    sum_1x2 = p_h_true + p_d_true + p_a_true
    p_h_true /= sum_1x2
    p_d_true /= sum_1x2
    p_a_true /= sum_1x2

    # Aplicar margen del bookmaker (ej. 6.5%) con algo de ruido/sesgo de mercado
    margin_mult = 1.065
    noise_h = random.uniform(0.96, 1.04)
    noise_a = random.uniform(0.96, 1.04)

    odds_h = max(1.10, round(1.0 / (p_h_true * margin_mult * noise_h), 2))
    odds_d = max(2.50, round(1.0 / (p_d_true * margin_mult), 2))
    odds_a = max(1.10, round(1.0 / (p_a_true * margin_mult * noise_a), 2))

    # 1X2 Snapshots
    for sel, val in [("home", odds_h), ("draw", odds_d), ("away", odds_a)]:
        session.add(
            OddsSnapshot(
                match_id=match.id,
                bookmaker="bet365",
                market=Market.ONE_X_TWO,
                selection=sel,
                odds_value=val,
                captured_at=match.scheduled_at - timedelta(minutes=15),
                is_closing_line=True,
            )
        )

    # Over / Under 2.5
    # Probabilidad aprox según media de goles
    p_over_2_5 = 1.0 - math_poisson_cdf(2, tot_lam)
    p_under_2_5 = 1.0 - p_over_2_5
    odds_o25 = max(1.15, round(1.0 / (max(0.08, p_over_2_5) * margin_mult * random.uniform(0.95, 1.05)), 2))
    odds_u25 = max(1.15, round(1.0 / (max(0.08, p_under_2_5) * margin_mult * random.uniform(0.95, 1.05)), 2))

    session.add(OddsSnapshot(match_id=match.id, bookmaker="bet365", market=Market.OVER_UNDER_2_5, selection="over", odds_value=odds_o25, captured_at=match.scheduled_at - timedelta(minutes=15), is_closing_line=True))
    session.add(OddsSnapshot(match_id=match.id, bookmaker="bet365", market=Market.OVER_UNDER_2_5, selection="under", odds_value=odds_u25, captured_at=match.scheduled_at - timedelta(minutes=15), is_closing_line=True))

    # BTTS (Both Teams To Score)
    p_btts = (1.0 - math_poisson_pmf(0, lam_h)) * (1.0 - math_poisson_pmf(0, lam_a))
    odds_btts_yes = max(1.15, round(1.0 / (max(0.10, p_btts) * margin_mult * random.uniform(0.95, 1.05)), 2))
    odds_btts_no = max(1.15, round(1.0 / (max(0.10, 1.0 - p_btts) * margin_mult * random.uniform(0.95, 1.05)), 2))

    session.add(OddsSnapshot(match_id=match.id, bookmaker="bet365", market=Market.BTTS, selection="yes", odds_value=odds_btts_yes, captured_at=match.scheduled_at - timedelta(minutes=15), is_closing_line=True))
    session.add(OddsSnapshot(match_id=match.id, bookmaker="bet365", market=Market.BTTS, selection="no", odds_value=odds_btts_no, captured_at=match.scheduled_at - timedelta(minutes=15), is_closing_line=True))


def math_poisson_pmf(k: int, lam: float) -> float:
    import math
    return (lam ** k * math.exp(-lam)) / math.factorial(k)


def math_poisson_cdf(k: int, lam: float) -> float:
    return sum(math_poisson_pmf(i, lam) for i in range(k + 1))
