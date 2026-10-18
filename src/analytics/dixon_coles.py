from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Sequence
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, poisson

from src.analytics.elo import DynamicEloEngine
from src.db.models import DurationFormat, Match


@dataclass
class MatchProbabilities:
    home_win: float
    draw: float
    away_win: float
    over_1_5: float
    under_1_5: float
    over_2_5: float
    under_2_5: float
    over_3_5: float
    under_3_5: float
    over_4_5: float
    under_4_5: float
    over_5_5: float
    under_5_5: float
    over_6_5: float
    under_6_5: float
    over_7_5: float
    under_7_5: float
    btts_yes: float
    btts_no: float
    expected_goals_home: float
    expected_goals_away: float
    expected_total_goals: float
    mle_weight: float                 # Peso asignado a MLE en el shrinkage (w_mle)
    score_matrix: np.ndarray          # Shape (max_goals+1, max_goals+1)
    top_scores: list[tuple[int, int, float]]

    def get_over_prob(self, line: float) -> float:
        """Calcula la probabilidad exacta de superar cualquier línea de goles."""
        matrix_dim = self.score_matrix.shape[0]
        x_coords, y_coords = np.indices((matrix_dim, matrix_dim))
        total_grid = x_coords + y_coords
        return float(np.sum(self.score_matrix[total_grid > line]))


@dataclass
class GammaLRTResult:
    statistic: float                  # 2 * (log_lik_unrestricted - log_lik_restricted)
    p_value: float                    # P-valor contra chi2(1)
    is_significant: bool              # True si p_value <= 0.05
    unrestricted_gamma: float         # Valor estimado de gamma
    decision: str                     # "retain_gamma" o "set_gamma_zero"


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Factor de corrección de Dixon & Coles para marcadores bajos.
    Ajusta la dependencia estocástica en marcadores {0-0, 1-0, 0-1, 1-1}.
    """
    if x == 0 and y == 0:
        return max(0.0001, 1.0 - lam * mu * rho)
    elif x == 0 and y == 1:
        return max(0.0001, 1.0 + lam * rho)
    elif x == 1 and y == 0:
        return max(0.0001, 1.0 + mu * rho)
    elif x == 1 and y == 1:
        return max(0.0001, 1.0 - rho)
    else:
        return 1.0


class DixonColesModel:
    """
    Motor estadístico Poisson Bivariado con corrección Dixon-Coles y Shrinkage Bayesiano.
    Entrenado de forma 100% aislada por DurationFormat (4x2, 5x2, 6x2).
    """

    def __init__(
        self,
        duration_format: DurationFormat,
        decay_xi: float = 0.005,
        window_size: int = 200,             # Ventana rodante (últimos W partidos)
        retrain_batch_size: int = 10,       # Frecuencia de reentrenamiento por lotes
        shrinkage_m: float = 5.0,           # Pseudo-cuentas m para el shrinkage con ELO
        force_gamma_zero: bool = False,     # Forzar gamma = 0 si no hay ventaja de saque
        auto_test_gamma: bool = True,       # Ejecutar LRT para decidir inclusión de gamma
        max_goals: int = 10,
    ):
        self.duration_format = duration_format
        self.decay_xi = decay_xi
        self.window_size = window_size
        self.retrain_batch_size = retrain_batch_size
        self.shrinkage_m = shrinkage_m
        self.force_gamma_zero = force_gamma_zero
        self.auto_test_gamma = auto_test_gamma
        self.max_goals = max_goals

        self.player_map: dict[int, int] = {}
        self.inv_player_map: dict[int, int] = {}
        self.attack_params: np.ndarray | None = None
        self.defense_params: np.ndarray | None = None
        self.home_advantage: float = 0.0
        self.rho: float = -0.04
        self.avg_goals: float = 2.2
        self.is_fitted: bool = False
        self.last_fitted_match_count: int = 0
        self.lrt_result: GammaLRTResult | None = None

    def test_gamma_significance(
        self,
        matches: Sequence[Match],
        as_of: datetime | None = None,
    ) -> GammaLRTResult:
        """
        Ejecuta el Test de Razón de Verosimilitud (Likelihood Ratio Test - LRT) para contrastar
        H0: gamma = 0 (sin ventaja de local/saque) vs H1: gamma != 0.
        """
        valid_matches = self._filter_and_window_matches(matches, as_of=as_of)
        if len(valid_matches) < 20:
            return GammaLRTResult(
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                unrestricted_gamma=0.0,
                decision="set_gamma_zero (muestra < 20)",
            )

        data = self._prepare_optimization_arrays(valid_matches, as_of=as_of)
        num_players = len(self.player_map)

        # 1. Ajuste no restringido (con gamma libre)
        res_unrestricted = self._optimize_mle(data, num_players, fit_gamma=True)
        log_lik_unrestricted = -res_unrestricted.fun
        gamma_hat = float(res_unrestricted.x[2 * num_players])

        # 2. Ajuste restringido (fijando gamma = 0)
        res_restricted = self._optimize_mle(data, num_players, fit_gamma=False)
        log_lik_restricted = -res_restricted.fun

        # 3. Estadístico de Razón de Verosimilitud: Lambda_LR = 2 * (ln L1 - ln L0)
        lr_stat = max(0.0, 2.0 * (log_lik_unrestricted - log_lik_restricted))
        p_val = float(chi2.sf(lr_stat, df=1))
        is_sig = (p_val <= 0.05)

        decision = "retain_gamma" if is_sig else "set_gamma_zero"
        lrt = GammaLRTResult(
            statistic=lr_stat,
            p_value=p_val,
            is_significant=is_sig,
            unrestricted_gamma=gamma_hat,
            decision=decision,
        )
        self.lrt_result = lrt
        return lrt

    def fit(
        self,
        matches: Sequence[Match],
        as_of: datetime | None = None,
        force_refit: bool = False,
    ) -> "DixonColesModel":
        """
        Ajusta el modelo MLE sobre la ventana rodante con decaimiento temporal.
        Aplica política de reentrenamiento por lotes (retrain_batch_size).
        """
        valid_matches = self._filter_and_window_matches(matches, as_of=as_of)
        n_matches = len(valid_matches)

        if n_matches < 5:
            self._fit_fallback_priors()
            return self

        # Política de lotes: evitar re-optimización costosa si no hay suficientes partidos nuevos
        if (
            self.is_fitted
            and not force_refit
            and abs(n_matches - self.last_fitted_match_count) < self.retrain_batch_size
        ):
            return self

        # Validación estadística de gamma si está habilitada
        if self.force_gamma_zero:
            fit_gamma = False
            self.home_advantage = 0.0
        elif self.auto_test_gamma:
            lrt = self.test_gamma_significance(valid_matches, as_of=as_of)
            fit_gamma = lrt.is_significant
        else:
            fit_gamma = True

        data = self._prepare_optimization_arrays(valid_matches, as_of=as_of)
        num_players = len(self.player_map)

        res = self._optimize_mle(data, num_players, fit_gamma=fit_gamma)

        if res.success:
            opt_p = res.x
            self.attack_params = opt_p[:num_players]
            self.defense_params = opt_p[num_players:2 * num_players]
            self.home_advantage = float(opt_p[2 * num_players]) if fit_gamma else 0.0
            self.rho = float(opt_p[2 * num_players + 1])
        else:
            self.attack_params = np.ones(num_players)
            self.defense_params = np.ones(num_players)
            self.home_advantage = 0.0
            self.rho = -0.04

        self.avg_goals = float(np.mean(data["h_goals"] + data["a_goals"])) / 2.0
        self.is_fitted = True
        self.last_fitted_match_count = n_matches
        return self

    def _filter_and_window_matches(
        self,
        matches: Sequence[Match],
        as_of: datetime | None = None,
    ) -> list[Match]:
        """Filtra por formato, fecha y recorta a la ventana rodante (rolling window)."""
        ref_date = as_of.replace(tzinfo=timezone.utc) if (as_of and as_of.tzinfo is None) else (as_of or datetime.now(timezone.utc))

        filtered = [
            m for m in matches
            if m.goals_home is not None
            and m.goals_away is not None
            and m.duration_format == self.duration_format
            and ((m.scheduled_at.replace(tzinfo=timezone.utc) if m.scheduled_at.tzinfo is None else m.scheduled_at) <= ref_date)
        ]

        # Ordenar cronológicamente y tomar los últimos W partidos
        filtered.sort(key=lambda m: m.scheduled_at.replace(tzinfo=timezone.utc) if m.scheduled_at.tzinfo is None else m.scheduled_at)
        if len(filtered) > self.window_size:
            filtered = filtered[-self.window_size:]

        return filtered

    def _prepare_optimization_arrays(
        self,
        valid_matches: list[Match],
        as_of: datetime | None = None,
    ) -> dict[str, np.ndarray]:
        """Indexa jugadores y genera vectores numpy optimizados para MLE."""
        player_ids = set()
        for m in valid_matches:
            player_ids.add(m.player_home_id)
            player_ids.add(m.player_away_id)

        player_list = sorted(list(player_ids))
        self.player_map = {pid: idx for idx, pid in enumerate(player_list)}
        self.inv_player_map = {idx: pid for idx, pid in enumerate(player_list)}

        ref_date = as_of.replace(tzinfo=timezone.utc) if (as_of and as_of.tzinfo is None) else (as_of or datetime.now(timezone.utc))

        h_idx, a_idx, h_g, a_g, w = [], [], [], [], []
        for m in valid_matches:
            h_idx.append(self.player_map[m.player_home_id])
            a_idx.append(self.player_map[m.player_away_id])
            h_g.append(m.goals_home)
            a_g.append(m.goals_away)

            m_dt = m.scheduled_at.replace(tzinfo=timezone.utc) if m.scheduled_at.tzinfo is None else m.scheduled_at
            days = max(0.0, (ref_date - m_dt).total_seconds() / 86400.0)
            w.append(math.exp(-self.decay_xi * days))

        return {
            "h_idx": np.array(h_idx, dtype=int),
            "a_idx": np.array(a_idx, dtype=int),
            "h_goals": np.array(h_g, dtype=int),
            "a_goals": np.array(a_g, dtype=int),
            "weights": np.array(w, dtype=float),
        }

    def _optimize_mle(
        self,
        data: dict[str, np.ndarray],
        num_players: int,
        fit_gamma: bool = True,
    ):
        """Minimiza la función de log-verosimilitud negativa con penalización L2."""
        h_idx = data["h_idx"]
        a_idx = data["a_idx"]
        h_g = data["h_goals"]
        a_g = data["a_goals"]
        w = data["weights"]

        init_params = np.concatenate([
            np.ones(num_players),
            np.ones(num_players),
            [0.0 if fit_gamma else 0.0, -0.04],
        ])

        def neg_log_lik(params: np.ndarray) -> float:
            alphas = params[:num_players]
            betas = params[num_players:2 * num_players]
            gamma_val = params[2 * num_players] if fit_gamma else 0.0
            rho_val = params[2 * num_players + 1]

            constraint_pen = 100.0 * (np.mean(alphas) - 1.0) ** 2

            lam = np.exp(alphas[h_idx] + betas[a_idx] + gamma_val)
            mu = np.exp(alphas[a_idx] + betas[h_idx])
            lam = np.clip(lam, 0.05, 20.0)
            mu = np.clip(mu, 0.05, 20.0)

            tau_vals = np.ones(len(h_idx))
            mask_00 = (h_g == 0) & (a_g == 0)
            mask_01 = (h_g == 0) & (a_g == 1)
            mask_10 = (h_g == 1) & (a_g == 0)
            mask_11 = (h_g == 1) & (a_g == 1)

            tau_vals[mask_00] = 1.0 - lam[mask_00] * mu[mask_00] * rho_val
            tau_vals[mask_01] = 1.0 + lam[mask_01] * rho_val
            tau_vals[mask_10] = 1.0 + mu[mask_10] * rho_val
            tau_vals[mask_11] = 1.0 - rho_val
            tau_vals = np.clip(tau_vals, 0.0001, 10.0)

            log_p_home = poisson.logpmf(h_g, lam)
            log_p_away = poisson.logpmf(a_g, mu)

            log_likelihood = np.sum(w * (np.log(tau_vals) + log_p_home + log_p_away))
            l2_reg = 0.05 * (np.sum((alphas - 1.0) ** 2) + np.sum((betas - 1.0) ** 2))

            return -(log_likelihood - l2_reg - constraint_pen)

        gamma_bounds = (-0.5, 0.5) if fit_gamma else (0.0, 0.0)
        bounds = (
            [(0.01, 4.0)] * num_players +
            [(0.01, 4.0)] * num_players +
            [gamma_bounds] +
            [(-0.25, 0.25)]
        )

        return minimize(
            neg_log_lik,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 150}
        )

    def _fit_fallback_priors(self):
        """Fallback para inicialización o muestra cero."""
        base_goals = {
            DurationFormat.MIN_8: 1.75,
            DurationFormat.MIN_10: 2.35,
            DurationFormat.MIN_12: 3.10,
        }.get(self.duration_format, 2.2)
        self.avg_goals = base_goals
        self.home_advantage = 0.0
        self.rho = -0.04
        self.is_fitted = True

    def calculate_lambdas(
        self,
        player_home_id: int,
        player_away_id: int,
        elo_home: float = 1500.0,
        elo_away: float = 1500.0,
        h2h_sample_size: int = 0,
    ) -> tuple[float, float, float]:
        """
        Calcula las tasas esperadas lambda y mu combinando Dixon-Coles MLE y el Prior ELO
        mediante Shrinkage Bayesiano explícito:
        w_mle = n / (n + m),  w_elo = m / (n + m)
        lambda_blended = w_mle * lambda_mle + w_elo * lambda_elo
        """
        has_home_mle = (self.is_fitted and self.attack_params is not None and player_home_id in self.player_map)
        has_away_mle = (self.is_fitted and self.attack_params is not None and player_away_id in self.player_map)

        if has_home_mle and has_away_mle:
            h_idx = self.player_map[player_home_id]
            a_idx = self.player_map[player_away_id]
            alpha_h = self.attack_params[h_idx]
            beta_h = self.defense_params[h_idx]
            alpha_a = self.attack_params[a_idx]
            beta_a = self.defense_params[a_idx]

            lam_mle = float(np.exp(alpha_h + beta_a + self.home_advantage))
            mu_mle = float(np.exp(alpha_a + beta_h))
        else:
            lam_mle = self.avg_goals
            mu_mle = self.avg_goals

        # Expectativa del prior ELO
        elo_engine = DynamicEloEngine()
        lam_elo, mu_elo = elo_engine.estimate_elo_goal_intensities(
            rating_home=elo_home,
            rating_away=elo_away,
            duration_format=self.duration_format,
            gamma=self.home_advantage,
        )

        # Si falta historial MLE de alguno, n=0 para usar 100% prior ELO
        effective_n = h2h_sample_size if (has_home_mle and has_away_mle) else 0

        lam_blended, mu_blended, w_mle = DynamicEloEngine.blend_goal_intensities(
            lam_mle=lam_mle,
            mu_mle=mu_mle,
            lam_elo=lam_elo,
            mu_elo=mu_elo,
            h2h_sample_size=effective_n,
            shrinkage_m=self.shrinkage_m,
        )

        return lam_blended, mu_blended, w_mle

    def predict_match(
        self,
        player_home_id: int,
        player_away_id: int,
        elo_home: float = 1500.0,
        elo_away: float = 1500.0,
        h2h_sample_size: int = 0,
        team_home: str | None = None,
        team_away: str | None = None,
    ) -> MatchProbabilities:
        """
        Genera la matriz de probabilidades bivariada y las probabilidades para todos los mercados,
        incorporando el nivel del gamer y el ajuste de tier del equipo EA Sports FC.
        """
        lam, mu, w_mle = self.calculate_lambdas(
            player_home_id=player_home_id,
            player_away_id=player_away_id,
            elo_home=elo_home,
            elo_away=elo_away,
            h2h_sample_size=h2h_sample_size,
        )

        # Ajuste por calidad del club/equipo (licencias EA Sports FC)
        if team_home or team_away:
            try:
                from src.ingestion.esportsbattle_scraper import get_team_tier_modifier
                mod_h = get_team_tier_modifier(team_home)
                mod_a = get_team_tier_modifier(team_away)
                lam = max(0.1, lam + mod_h - (mod_a * 0.5))
                mu = max(0.1, mu + mod_a - (mod_h * 0.5))
            except Exception:
                pass

        matrix_dim = self.max_goals + 1
        score_matrix = np.zeros((matrix_dim, matrix_dim), dtype=float)

        p_home_goals = [poisson.pmf(g, lam) for g in range(matrix_dim)]
        p_away_goals = [poisson.pmf(g, mu) for g in range(matrix_dim)]

        for x in range(matrix_dim):
            for y in range(matrix_dim):
                tau = dixon_coles_tau(x, y, lam, mu, self.rho)
                score_matrix[x, y] = tau * p_home_goals[x] * p_away_goals[y]

        mat_sum = np.sum(score_matrix)
        if mat_sum > 0:
            score_matrix /= mat_sum

        # Mercados 1X2
        home_win_p = float(np.sum(np.tril(score_matrix, -1)))
        draw_p = float(np.sum(np.diag(score_matrix)))
        away_win_p = float(np.sum(np.triu(score_matrix, 1)))

        # Mercados Totales Over / Under
        x_coords, y_coords = np.indices((matrix_dim, matrix_dim))
        total_grid = x_coords + y_coords

        over_1_5_p = float(np.sum(score_matrix[total_grid > 1.5]))
        over_2_5_p = float(np.sum(score_matrix[total_grid > 2.5]))
        over_3_5_p = float(np.sum(score_matrix[total_grid > 3.5]))
        over_4_5_p = float(np.sum(score_matrix[total_grid > 4.5]))
        over_5_5_p = float(np.sum(score_matrix[total_grid > 5.5]))
        over_6_5_p = float(np.sum(score_matrix[total_grid > 6.5]))
        over_7_5_p = float(np.sum(score_matrix[total_grid > 7.5]))

        # Mercado Ambos Marcan (BTTS)
        btts_yes_p = float(np.sum(score_matrix[1:, 1:]))

        # Top marcadores
        top_scores: list[tuple[int, int, float]] = []
        flat_indices = np.argsort(score_matrix.flatten())[::-1][:10]
        for idx in flat_indices:
            x_g, y_g = divmod(int(idx), matrix_dim)
            top_scores.append((x_g, y_g, float(score_matrix[x_g, y_g])))

        return MatchProbabilities(
            home_win=home_win_p,
            draw=draw_p,
            away_win=away_win_p,
            over_1_5=over_1_5_p,
            under_1_5=1.0 - over_1_5_p,
            over_2_5=over_2_5_p,
            under_2_5=1.0 - over_2_5_p,
            over_3_5=over_3_5_p,
            under_3_5=1.0 - over_3_5_p,
            over_4_5=over_4_5_p,
            under_4_5=1.0 - over_4_5_p,
            over_5_5=over_5_5_p,
            under_5_5=1.0 - over_5_5_p,
            over_6_5=over_6_5_p,
            under_6_5=1.0 - over_6_5_p,
            over_7_5=over_7_5_p,
            under_7_5=1.0 - over_7_5_p,
            btts_yes=btts_yes_p,
            btts_no=1.0 - btts_yes_p,
            expected_goals_home=lam,
            expected_goals_away=mu,
            expected_total_goals=lam + mu,
            mle_weight=w_mle,
            score_matrix=score_matrix,
            top_scores=top_scores,
        )
