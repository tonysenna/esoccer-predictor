from __future__ import annotations

import math
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from src.db.models import DurationFormat, Match, RatingHistory
from src.db.queries import get_latest_player_rating


class DynamicEloEngine:
    """
    Sistema de rating dinámico ELO / Glicko adaptado a e-Soccer / FIFA Esports.

    Características clave:
    1. Segmentación Estricta: Cada formato de duración (4x2, 5x2, 6x2) mantiene
       una serie temporal de ratings totalmente independiente.
    2. Multiplicador por Margen de Goles (GDM): Escala logarítmica para calibrar
       victorias contundentes sin distorsión por outliers.
    3. Rating Deviation (RD): Reduce la incertidumbre a medida que aumenta la muestra
       de partidos jugados por el gamer.
    4. Estimador de Tasas de Gol (Lambda/Mu Prior): Convierte la diferencia de rating
       a tasas de gol esperadas para actuar como Prior Bayesiano.
    """

    def __init__(
        self,
        base_rating: float = 1500.0,
        base_rd: float = 350.0,
        k_factor: float = 32.0,
        goal_scale: float = 0.45,
        inactivity_c: float = 51.12,  # Constante Glicko c: 45 días de inactividad devuelven RD de 70.0 a 350.0 exacto
    ):
        self.base_rating = base_rating
        self.base_rd = base_rd
        self.k_factor = k_factor
        self.goal_scale = goal_scale
        self.inactivity_c = inactivity_c

    def get_player_rating(
        self,
        session: Session,
        player_id: int,
        duration_format: DurationFormat,
        as_of: datetime | None = None,
    ) -> tuple[float, float, int]:
        """
        Retorna (rating, rating_deviation, matches_played) para un gamer en un formato.
        Si existe historial, la incertidumbre (RD) se incrementa por inactividad (Glicko step 1):
        RD_actual = min(base_rd, sqrt(RD_prev^2 + c^2 * elapsed_days))
        Si as_of es None, se toma datetime.now(timezone.utc) por defecto.
        """
        latest = get_latest_player_rating(session, player_id, duration_format, before_date=as_of)
        if latest:
            r = float(latest.rating)
            rd = float(latest.rating_deviation)
            n = int(latest.matches_played)

            if latest.as_of:
                ref_t = as_of.replace(tzinfo=timezone.utc) if (as_of and as_of.tzinfo is None) else (as_of or datetime.now(timezone.utc))
                prev_t = latest.as_of.replace(tzinfo=timezone.utc) if latest.as_of.tzinfo is None else latest.as_of
                elapsed_days = max(0.0, (ref_t - prev_t).total_seconds() / 86400.0)
                if elapsed_days > 0:
                    rd = min(self.base_rd, math.sqrt(rd ** 2 + (self.inactivity_c ** 2) * elapsed_days))

            return r, rd, n
        return self.base_rating, self.base_rd, 0

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probabilidad esperada de victoria para el jugador A frente a B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def goal_difference_multiplier(self, goals_a: int, goals_b: int, rating_diff: float) -> float:
        """
        Multiplicador por margen de victoria en goles.
        GDM = 1.0 si diferencia <= 1 gol.
        GDM = 1.5 si diferencia == 2 goles.
        GDM = (1.75 + (diff - 3) * 0.25) * (1 + exp(-|diff_elo|/400)*0.2) para >= 3 goles.
        """
        diff = abs(goals_a - goals_b)
        if diff <= 1:
            return 1.0
        elif diff == 2:
            return 1.5
        else:
            return (1.75 + (diff - 3) * 0.25) * (1.0 + math.exp(-abs(rating_diff) / 400.0) * 0.2)

    def update_ratings_for_match(
        self,
        session: Session,
        match: Match,
        commit: bool = False,
    ) -> tuple[RatingHistory, RatingHistory]:
        """
        Actualiza y persiste en `RatingHistory` la evolución del rating de ambos gamers
        tras un partido finalizado.
        """
        if match.goals_home is None or match.goals_away is None:
            raise ValueError(f"Match {match.id} no tiene resultado de goles")

        fmt = match.duration_format
        match_time = match.finished_at or match.scheduled_at or datetime.now(timezone.utc)

        r_home, rd_home, n_home = self.get_player_rating(session, match.player_home_id, fmt, as_of=match_time)
        r_away, rd_away, n_away = self.get_player_rating(session, match.player_away_id, fmt, as_of=match_time)

        # Resultado real (1: Home win, 0.5: Draw, 0: Away win)
        if match.goals_home > match.goals_away:
            s_home, s_away = 1.0, 0.0
        elif match.goals_home == match.goals_away:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        # Expectativa ELO
        e_home = self.expected_score(r_home, r_away)
        e_away = 1.0 - e_home

        # Multiplicador por diferencia de goles
        gdm = self.goal_difference_multiplier(match.goals_home, match.goals_away, r_home - r_away)

        # Factor K modulado por la incertidumbre (Rating Deviation)
        k_home = self.k_factor * (1.0 + (rd_home / self.base_rd) * 0.5)
        k_away = self.k_factor * (1.0 + (rd_away / self.base_rd) * 0.5)

        new_r_home = r_home + k_home * gdm * (s_home - e_home)
        new_r_away = r_away + k_away * gdm * (s_away - e_away)

        # Reducción de desviación por experiencia acumulada
        new_rd_home = max(50.0, rd_home * 0.96)
        new_rd_away = max(50.0, rd_away * 0.96)

        hist_home = RatingHistory(
            player_id=match.player_home_id,
            duration_format=fmt,
            rating=new_r_home,
            rating_deviation=new_rd_home,
            matches_played=n_home + 1,
            as_of=match_time,
        )
        hist_away = RatingHistory(
            player_id=match.player_away_id,
            duration_format=fmt,
            rating=new_r_away,
            rating_deviation=new_rd_away,
            matches_played=n_away + 1,
            as_of=match_time,
        )

        session.add(hist_home)
        session.add(hist_away)

        if commit:
            session.commit()
        else:
            session.flush()

        return hist_home, hist_away

    def estimate_elo_goal_intensities(
        self,
        rating_home: float,
        rating_away: float,
        duration_format: DurationFormat,
        gamma: float = 0.0,
    ) -> tuple[float, float]:
        """
        Convierte la diferencia de rating ELO a intensidades esperadas de gol (lambda_elo, mu_elo)
        calibradas al ritmo de cada formato.

        Fórmulas:
        lambda_elo = G_bar * exp( ((R_h - R_a)/400) * c_elo + gamma )
        mu_elo     = G_bar * exp( -((R_h - R_a)/400) * c_elo )
        """
        base_goals_per_format = {
            DurationFormat.MIN_8: 1.75,   # ~3.5 goles totales
            DurationFormat.MIN_10: 2.35,  # ~4.7 goles totales
            DurationFormat.MIN_12: 3.10,  # ~6.2 goles totales
        }
        avg_gamer_goals = base_goals_per_format.get(duration_format, 2.2)

        diff = rating_home - rating_away
        factor = (diff / 400.0) * self.goal_scale

        lambda_home = max(0.15, avg_gamer_goals * math.exp(factor + gamma))
        lambda_away = max(0.15, avg_gamer_goals * math.exp(-factor))

        return lambda_home, lambda_away

    @staticmethod
    def blend_goal_intensities(
        lam_mle: float,
        mu_mle: float,
        lam_elo: float,
        mu_elo: float,
        h2h_sample_size: int,
        shrinkage_m: float = 5.0,
    ) -> tuple[float, float, float]:
        """
        Fórmula de Shrinkage Bayesiano para combinar Dixon-Coles MLE con el Prior ELO.

        Definición:
        Sea n = h2h_sample_size (número de partidos previos en ese formato).
        Sea m = shrinkage_m (intensidad del prior / pseudo-cuentas, default m = 5.0).

        Pesos de combinación:
        w_mle = n / (n + m)
        w_elo = m / (n + m) = 1 - w_mle

        Intensidades combinadas:
        lambda_blended = w_mle * lambda_mle + w_elo * lambda_elo
        mu_blended     = w_mle * mu_mle     + w_elo * mu_elo

        Retorna:
        (lambda_blended, mu_blended, w_mle)
        """
        n = max(0, h2h_sample_size)
        m = max(0.1, shrinkage_m)

        w_mle = n / (n + m)
        w_elo = m / (n + m)

        lam_blended = w_mle * lam_mle + w_elo * lam_elo
        mu_blended = w_mle * mu_mle + w_elo * mu_elo

        # Guardas de cota para e-Soccer
        lam_blended = max(0.15, min(lam_blended, 12.0))
        mu_blended = max(0.15, min(mu_blended, 12.0))

        return lam_blended, mu_blended, w_mle
