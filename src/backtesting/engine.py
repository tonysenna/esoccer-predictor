from __future__ import annotations

from datetime import datetime
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.analytics.dixon_coles import DixonColesModel
from src.analytics.elo import DynamicEloEngine
from src.analytics.odds import evaluate_market_value
from src.backtesting.metrics import (
    BacktestBetLog,
    BacktestReport,
    FormatPerformance,
    MarketPerformance,
    calculate_brier_score_binary,
    calculate_brier_score_multiclass,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)
from src.db.models import DurationFormat, Market, Match, MatchStatus, OddsSnapshot, Player
from src.db.queries import get_h2h_matches


class BacktestEngine:
    """
    Motor de Backtesting Walk-Forward para e-Soccer / FIFA Esports.
    Garantiza una evaluación honesta y sin sesgo de anticipación (lookahead bias).
    """

    def __init__(
        self,
        session: Session,
        initial_bankroll: float = 1000.0,
        min_edge_threshold: float = 0.03,  # 3% edge mínimo
        stake_mode: str = "kelly",         # "kelly" o "flat"
        flat_stake_amount: float = 20.0,
        kelly_fraction: float = 0.25,      # Cuarto de Kelly
        max_stake_percent: float = 0.05,   # Máximo 5% de la banca actual
        decay_xi: float = 0.005,
    ):
        self.session = session
        self.initial_bankroll = initial_bankroll
        self.min_edge_threshold = min_edge_threshold
        self.stake_mode = stake_mode
        self.flat_stake_amount = flat_stake_amount
        self.kelly_fraction = kelly_fraction
        self.max_stake_percent = max_stake_percent
        self.decay_xi = decay_xi

        self.elo_engine = DynamicEloEngine()
        # Modelos independientes por formato de duración
        self.models: dict[DurationFormat, DixonColesModel] = {
            fmt: DixonColesModel(duration_format=fmt, decay_xi=decay_xi)
            for fmt in DurationFormat
        }

    def run(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        duration_format: DurationFormat | None = None,
        min_training_matches: int = 15,
    ) -> BacktestReport:
        """
        Ejecuta la simulación histórica completa paso a paso.
        """
        # 1. Cargar todos los partidos finalizados en orden cronológico
        stmt = (
            select(Match)
            .where(
                Match.status == MatchStatus.FINISHED,
                Match.goals_home.is_not(None),
                Match.goals_away.is_not(None),
            )
            .order_by(Match.scheduled_at.asc())
        )
        if duration_format is not None:
            stmt = stmt.where(Match.duration_format == duration_format)

        all_matches = list(self.session.execute(stmt).scalars().all())

        if not all_matches:
            return BacktestReport(
                total_matches_evaluated=0,
                total_bets_placed=0,
                won_bets=0,
                win_rate=0.0,
                total_staked=0.0,
                total_profit=0.0,
                roi=0.0,
                final_bankroll=self.initial_bankroll,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                global_brier_score_1x2=0.0,
                global_brier_score_totals=0.0,
                log_loss_1x2=0.0,
            )

        current_bankroll = self.initial_bankroll
        bankroll_history: list[float] = [current_bankroll]
        bet_logs: list[BacktestBetLog] = []
        pnl_per_bet: list[float] = []

        # Métricas de calibración
        mc_1x2_probs: list[list[float]] = []
        mc_1x2_actuals: list[int] = []
        binary_totals_probs: list[float] = []
        binary_totals_actuals: list[int] = []

        # Seguimiento por mercado
        market_stats: dict[str, dict[str, float]] = {}
        format_stats: dict[str, dict[str, float]] = {}

        # Mapeo rápido de nombres de jugadores
        players_stmt = select(Player)
        players = {p.id: p.display_name for p in self.session.execute(players_stmt).scalars().all()}

        # Partidos procesados históricamente por formato (para entrenamiento)
        history_by_format: dict[DurationFormat, list[Match]] = {
            fmt: [] for fmt in DurationFormat
        }

        # Iterar cronológicamente
        for i, match in enumerate(all_matches):
            fmt = match.duration_format
            match_date = match.scheduled_at

            # Verificar si está en la ventana de evaluación
            in_eval_window = True
            if start_date and match_date < start_date:
                in_eval_window = False
            if end_date and match_date > end_date:
                in_eval_window = False

            fmt_history = history_by_format[fmt]

            # Si ya tenemos suficiente historial y estamos dentro de la ventana de backtest:
            if in_eval_window and len(fmt_history) >= min_training_matches:
                # 1. Ajustar modelo Dixon-Coles sólo con datos pasados
                model = self.models[fmt]
                model.fit(fmt_history, as_of=match_date)

                # 2. Obtener ratings ELO previos
                r_home, _, _ = self.elo_engine.get_player_rating(self.session, match.player_home_id, fmt, as_of=match_date)
                r_away, _, _ = self.elo_engine.get_player_rating(self.session, match.player_away_id, fmt, as_of=match_date)

                # 3. Calcular tamaño muestral H2H previo (sin lookahead bias)
                h2h_prior_matches = get_h2h_matches(
                    session=self.session,
                    player_a_id=match.player_home_id,
                    player_b_id=match.player_away_id,
                    duration_format=fmt,
                    before_date=match_date,
                    limit=50,
                )
                h2h_n = len(h2h_prior_matches)

                # 4. Predecir probabilidades combinando Dixon-Coles y Prior ELO via Shrinkage
                probs = model.predict_match(
                    player_home_id=match.player_home_id,
                    player_away_id=match.player_away_id,
                    elo_home=r_home,
                    elo_away=r_away,
                    h2h_sample_size=h2h_n,
                )

                # Registrar para Brier Score 1X2
                mc_1x2_probs.append([probs.home_win, probs.draw, probs.away_win])
                if match.goals_home > match.goals_away:
                    actual_1x2_idx = 0
                    actual_1x2_str = "home"
                elif match.goals_home == match.goals_away:
                    actual_1x2_idx = 1
                    actual_1x2_str = "draw"
                else:
                    actual_1x2_idx = 2
                    actual_1x2_str = "away"
                mc_1x2_actuals.append(actual_1x2_idx)

                # Registrar para Brier Score Totals Over 2.5
                tot_goals = match.goals_home + match.goals_away
                binary_totals_probs.append(probs.over_2_5)
                binary_totals_actuals.append(1 if tot_goals > 2.5 else 0)

                # 4. Obtener cuotas de cierre disponibles
                odds_stmt = select(OddsSnapshot).where(OddsSnapshot.match_id == match.id)
                match_odds = list(self.session.execute(odds_stmt).scalars().all())

                # Agrupar cuotas por mercado
                odds_by_market: dict[Market, dict[str, float]] = {}
                for o in match_odds:
                    if o.market not in odds_by_market:
                        odds_by_market[o.market] = {}
                    odds_by_market[o.market][o.selection] = o.odds_value

                # 5. Evaluar mercados para detectar +EV
                self._evaluate_and_place_bets(
                    match=match,
                    probs=probs,
                    odds_by_market=odds_by_market,
                    current_bankroll=current_bankroll,
                    players=players,
                    bet_logs=bet_logs,
                    pnl_per_bet=pnl_per_bet,
                    bankroll_history=bankroll_history,
                    market_stats=market_stats,
                    format_stats=format_stats,
                )
                if bankroll_history:
                    current_bankroll = bankroll_history[-1]

            # 6. Actualizar historial y ratings ELO para el siguiente paso del tiempo
            history_by_format[fmt].append(match)
            self.elo_engine.update_ratings_for_match(self.session, match, commit=False)

        # Compilar reporte final
        total_bets = len(bet_logs)
        won_bets = sum(1 for b in bet_logs if b.is_won)
        win_rate = (won_bets / total_bets * 100.0) if total_bets > 0 else 0.0
        total_staked = sum(b.stake for b in bet_logs)
        total_profit = sum(b.pnl for b in bet_logs)
        roi = (total_profit / total_staked * 100.0) if total_staked > 0 else 0.0

        brier_1x2 = calculate_brier_score_multiclass(np.array(mc_1x2_probs), mc_1x2_actuals) if mc_1x2_probs else 0.0
        brier_totals = calculate_brier_score_binary(binary_totals_probs, binary_totals_actuals) if binary_totals_probs else 0.0
        max_dd = calculate_max_drawdown(bankroll_history)
        sharpe = calculate_sharpe_ratio(pnl_per_bet)

        # Breakdown de mercados
        market_reports: dict[str, MarketPerformance] = {}
        for m_name, st in market_stats.items():
            t_b = int(st["bets"])
            w_b = int(st["wins"])
            t_stk = st["staked"]
            t_pnl = st["pnl"]
            market_reports[m_name] = MarketPerformance(
                market=m_name,
                total_bets=t_b,
                won_bets=w_b,
                win_rate=(w_b / t_b * 100.0) if t_b > 0 else 0.0,
                total_staked=t_stk,
                total_profit=t_pnl,
                roi=(t_pnl / t_stk * 100.0) if t_stk > 0 else 0.0,
                avg_odds=st["odds_sum"] / t_b if t_b > 0 else 0.0,
                avg_edge=st["edge_sum"] / t_b if t_b > 0 else 0.0,
                brier_score=0.0,
            )

        # Breakdown de formatos de duración
        format_reports: dict[str, FormatPerformance] = {}
        for f_name, st in format_stats.items():
            t_b = int(st["bets"])
            w_b = int(st["wins"])
            t_stk = st["staked"]
            t_pnl = st["pnl"]
            format_reports[f_name] = FormatPerformance(
                duration_format=f_name,
                total_bets=t_b,
                won_bets=w_b,
                win_rate=(w_b / t_b * 100.0) if t_b > 0 else 0.0,
                total_staked=t_stk,
                total_profit=t_pnl,
                roi=(t_pnl / t_stk * 100.0) if t_stk > 0 else 0.0,
                avg_odds=st["odds_sum"] / t_b if t_b > 0 else 0.0,
                avg_edge=st["edge_sum"] / t_b if t_b > 0 else 0.0,
            )

        return BacktestReport(
            total_matches_evaluated=len(all_matches),
            total_bets_placed=total_bets,
            won_bets=won_bets,
            win_rate=win_rate,
            total_staked=total_staked,
            total_profit=total_profit,
            roi=roi,
            final_bankroll=current_bankroll,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            global_brier_score_1x2=brier_1x2,
            global_brier_score_totals=brier_totals,
            log_loss_1x2=0.0,
            market_breakdown=market_reports,
            format_breakdown=format_reports,
            bet_history=bet_logs,
            pnl_series=bankroll_history,
        )

    def _evaluate_and_place_bets(
        self,
        match: Match,
        probs,
        odds_by_market: dict[Market, dict[str, float]],
        current_bankroll: float,
        players: dict[int, str],
        bet_logs: list[BacktestBetLog],
        pnl_per_bet: list[float],
        bankroll_history: list[float],
        market_stats: dict[str, dict[str, float]],
        format_stats: dict[str, dict[str, float]],
    ):
        """Evalúa las oportunidades de valor y simula la colocación de apuestas."""
        tot_goals = match.goals_home + match.goals_away
        fmt_str = match.duration_format.value

        # Definir probabilidades calculadas por mercado
        market_model_probs = {
            Market.ONE_X_TWO: {"home": probs.home_win, "draw": probs.draw, "away": probs.away_win},
            Market.OVER_UNDER_1_5: {"over": probs.over_1_5, "under": probs.under_1_5},
            Market.OVER_UNDER_2_5: {"over": probs.over_2_5, "under": probs.under_2_5},
            Market.OVER_UNDER_3_5: {"over": probs.over_3_5, "under": probs.under_3_5},
            Market.OVER_UNDER_4_5: {"over": probs.over_4_5, "under": probs.under_4_5},
            Market.BTTS: {"yes": probs.btts_yes, "no": probs.btts_no},
        }

        # Determinación de resultados reales
        actual_results = {
            Market.ONE_X_TWO: "home" if match.goals_home > match.goals_away else ("draw" if match.goals_home == match.goals_away else "away"),
            Market.OVER_UNDER_1_5: "over" if tot_goals > 1.5 else "under",
            Market.OVER_UNDER_2_5: "over" if tot_goals > 2.5 else "under",
            Market.OVER_UNDER_3_5: "over" if tot_goals > 3.5 else "under",
            Market.OVER_UNDER_4_5: "over" if tot_goals > 4.5 else "under",
            Market.BTTS: "yes" if (match.goals_home > 0 and match.goals_away > 0) else "no",
        }

        for mkt, odds_dict in odds_by_market.items():
            if mkt not in market_model_probs:
                continue

            m_probs = market_model_probs[mkt]
            # Validar que tengamos cuotas para todas las opciones del mercado
            if not all(k in odds_dict for k in m_probs.keys()):
                continue

            value_bets = evaluate_market_value(
                market=mkt,
                selections_odds=odds_dict,
                model_probs=m_probs,
                min_edge=self.min_edge_threshold,
                kelly_fraction=self.kelly_fraction,
                max_stake=self.max_stake_percent,
            )

            for vb in value_bets:
                if not vb.is_value:
                    continue

                # Determinar tamaño de apuesta (Stake)
                if self.stake_mode == "kelly":
                    stake = current_bankroll * vb.kelly_stake_pct
                else:
                    stake = min(self.flat_stake_amount, current_bankroll * self.max_stake_percent)

                if stake <= 0.5 or current_bankroll <= 5.0:
                    continue  # Banca insuficiente

                actual_sel = actual_results[mkt]
                is_won = (vb.selection == actual_sel)

                if is_won:
                    pnl = stake * (vb.decimal_odds - 1.0)
                else:
                    pnl = -stake

                current_bankroll += pnl
                bankroll_history.append(current_bankroll)
                pnl_per_bet.append(pnl)

                # Registrar en logs
                home_name = players.get(match.player_home_id, f"Gamer {match.player_home_id}")
                away_name = players.get(match.player_away_id, f"Gamer {match.player_away_id}")

                bet_logs.append(
                    BacktestBetLog(
                        match_id=match.id,
                        scheduled_at=match.scheduled_at.isoformat() if match.scheduled_at else "",
                        duration_format=fmt_str,
                        home_player=home_name,
                        away_player=away_name,
                        market=mkt.value,
                        selection=vb.selection,
                        odds=vb.decimal_odds,
                        model_prob=vb.model_prob,
                        fair_implied_prob=vb.fair_implied_prob,
                        ev_edge=vb.ev_edge,
                        stake=stake,
                        actual_result=actual_sel,
                        is_won=is_won,
                        pnl=pnl,
                        bankroll_after=current_bankroll,
                    )
                )

                # Acumular estadísticas de mercado
                m_key = mkt.value
                if m_key not in market_stats:
                    market_stats[m_key] = {"bets": 0, "wins": 0, "staked": 0.0, "pnl": 0.0, "odds_sum": 0.0, "edge_sum": 0.0}
                market_stats[m_key]["bets"] += 1
                if is_won:
                    market_stats[m_key]["wins"] += 1
                market_stats[m_key]["staked"] += stake
                market_stats[m_key]["pnl"] += pnl
                market_stats[m_key]["odds_sum"] += vb.decimal_odds
                market_stats[m_key]["edge_sum"] += vb.ev_edge

                # Acumular estadísticas de formato
                if fmt_str not in format_stats:
                    format_stats[fmt_str] = {"bets": 0, "wins": 0, "staked": 0.0, "pnl": 0.0, "odds_sum": 0.0, "edge_sum": 0.0}
                format_stats[fmt_str]["bets"] += 1
                if is_won:
                    format_stats[fmt_str]["wins"] += 1
                format_stats[fmt_str]["staked"] += stake
                format_stats[fmt_str]["pnl"] += pnl
                format_stats[fmt_str]["odds_sum"] += vb.decimal_odds
                format_stats[fmt_str]["edge_sum"] += vb.ev_edge
