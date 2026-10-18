from __future__ import annotations

from datetime import datetime, timezone
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.analytics.dixon_coles import DixonColesModel
from src.analytics.elo import DynamicEloEngine
from src.analytics.metrics import calculate_h2h_metrics, calculate_player_form
from src.analytics.odds import evaluate_market_value
from src.backtesting.engine import BacktestEngine
from src.config import get_settings
from src.db.models import DurationFormat, Market
from src.db.queries import (
    get_all_matches_for_format,
    get_h2h_matches,
    get_or_create_player,
    get_player_by_name,
    get_recent_form,
)
from src.db.session import SessionLocal, init_db
from src.ingestion.mock_generator import seed_realistic_esoccer_dataset

# Consola robusta para Windows y terminales estándar
console = Console(safe_box=True)
settings = get_settings()


def run_predict_cli(
    gamer_a_name: str,
    gamer_b_name: str,
    format_str: str = "5x2",
    odds_1x2: tuple[float, float, float] | None = None,
    odds_o25: tuple[float, float] | None = None,
    odds_btts: tuple[float, float] | None = None,
):
    """Ejecuta una predicción completa y análisis +EV entre Gamer A y Gamer B."""
    # Mapeo de formato de duración
    fmt_map = {
        "4x2": DurationFormat.MIN_8,
        "8": DurationFormat.MIN_8,
        "5x2": DurationFormat.MIN_10,
        "10": DurationFormat.MIN_10,
        "6x2": DurationFormat.MIN_12,
        "12": DurationFormat.MIN_12,
    }
    duration_fmt = fmt_map.get(format_str.lower(), DurationFormat.MIN_10)

    db = SessionLocal()
    try:
        player_a = get_player_by_name(db, gamer_a_name) or get_or_create_player(db, gamer_a_name)
        player_b = get_player_by_name(db, gamer_b_name) or get_or_create_player(db, gamer_b_name)

        # 1. Rating dinámico ELO
        elo_engine = DynamicEloEngine(
            base_rating=settings.BASE_ELO_RATING,
            k_factor=settings.ELO_K_FACTOR,
        )
        r_a, rd_a, n_a = elo_engine.get_player_rating(db, player_a.id, duration_fmt)
        r_b, rd_b, n_b = elo_engine.get_player_rating(db, player_b.id, duration_fmt)

        # 2. H2H y Formas Recientes (Segmentadas por formato)
        h2h_matches = get_h2h_matches(db, player_a.id, player_b.id, duration_format=duration_fmt, limit=20)
        h2h_stats = calculate_h2h_metrics(player_a.id, player_b.id, h2h_matches, duration_format=duration_fmt)

        form_a_matches = get_recent_form(db, player_a.id, duration_fmt, limit=settings.RECENT_MATCHES_COUNT)
        form_b_matches = get_recent_form(db, player_b.id, duration_fmt, limit=settings.RECENT_MATCHES_COUNT)

        form_a = calculate_player_form(player_a.id, form_a_matches, duration_fmt, decay_xi=0.05)
        form_b = calculate_player_form(player_b.id, form_b_matches, duration_fmt, decay_xi=0.05)

        # 3. Ajuste del modelo Dixon-Coles
        training_matches = get_all_matches_for_format(db, duration_fmt)
        dc_model = DixonColesModel(duration_format=duration_fmt, decay_xi=settings.TIME_DECAY_XI)
        dc_model.fit(training_matches)

        # 4. Predicción combinando Dixon-Coles y Prior ELO via Shrinkage
        pred = dc_model.predict_match(
            player_home_id=player_a.id,
            player_away_id=player_b.id,
            elo_home=r_a,
            elo_away=r_b,
            h2h_sample_size=h2h_stats.sample_size,
        )

        # RENDERIZADO VISUAL
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]e-Soccer Predictor & Value Engine[/bold cyan]\n"
            f"[bold white]{player_a.display_name} (Local)[/bold white] vs [bold white]{player_b.display_name} (Visitante)[/bold white]\n"
            f"Formato: [bold yellow]{duration_fmt.value} ({'8 min' if duration_fmt == DurationFormat.MIN_8 else ('10 min' if duration_fmt == DurationFormat.MIN_10 else '12 min')})[/bold yellow]\n"
            f"Shrinkage Bayesiano: [bold green]Dixon-Coles MLE ({pred.mle_weight * 100:.1f}%)[/bold green] | [bold blue]Prior ELO ({(1.0 - pred.mle_weight) * 100:.1f}%)[/bold blue] (n={h2h_stats.sample_size}, m=5.0)",
            border_style="cyan"
        ))

        # Métricas y Ratings
        metrics_table = Table(title="[METRICAS] Metricas de Base & Rating Dinamico", show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metrica", style="dim")
        metrics_table.add_column(f"{player_a.display_name} (A)", justify="center")
        metrics_table.add_column(f"{player_b.display_name} (B)", justify="center")
        metrics_table.add_column("Contexto H2H", justify="center")

        h2h_desc = (
            f"[green]{h2h_stats.a_wins}W[/green] - {h2h_stats.draws}D - [red]{h2h_stats.b_wins}W[/red] ({h2h_stats.sample_size} p.)"
            if h2h_stats.is_sufficient_sample
            else f"[yellow]Muestra baja ({h2h_stats.sample_size} p. < 5)[/yellow]\n[dim]Respaldo ELO activo[/dim]"
        )

        metrics_table.add_row(
            "Rating Dinamico ELO",
            f"[bold]{r_a:.1f}[/bold] (+/-{rd_a:.0f})",
            f"[bold]{r_b:.1f}[/bold] (+/-{rd_b:.0f})",
            f"Diff ELO: {r_a - r_b:+.1f}"
        )
        metrics_table.add_row(
            "Goles Ponderados F / C",
            f"{form_a.weighted_goals_for:.2f} / {form_a.weighted_goals_against:.2f}",
            f"{form_b.weighted_goals_for:.2f} / {form_b.weighted_goals_against:.2f}",
            f"Media H2H: {h2h_stats.total_goals_avg:.1f} g/p"
        )
        metrics_table.add_row(
            "Racha % Over 2.5",
            f"{form_a.over_2_5_rate * 100:.1f}%",
            f"{form_b.over_2_5_rate * 100:.1f}%",
            f"H2H Over 2.5: {h2h_stats.over_2_5_rate * 100:.1f}%"
        )
        metrics_table.add_row(
            "Racha % Ambos Marcan",
            f"{form_a.btts_rate * 100:.1f}%",
            f"{form_b.btts_rate * 100:.1f}%",
            f"H2H BTTS: {h2h_stats.btts_rate * 100:.1f}%"
        )
        console.print(metrics_table)

        # Probabilidades Dixon-Coles
        prob_table = Table(title="[PROBABILIDADES] Dixon-Coles Bivariadas", show_header=True, header_style="bold green")
        prob_table.add_column("Mercado", style="bold")
        prob_table.add_column("Seleccion", justify="center")
        prob_table.add_column("Prob. Calculada", justify="center", style="bold cyan")
        prob_table.add_column("Cuota Justa (Fair Odds)", justify="center", style="bold yellow")

        def fair_odds(p: float) -> str:
            return f"@{1.0 / p:.2f}" if p > 0 else "-"

        # 1X2
        prob_table.add_row("1X2 (Resultado Final)", f"1 ({player_a.display_name})", f"{pred.home_win * 100:.1f}%", fair_odds(pred.home_win))
        prob_table.add_row("", "X (Empate)", f"{pred.draw * 100:.1f}%", fair_odds(pred.draw))
        prob_table.add_row("", f"2 ({player_b.display_name})", f"{pred.away_win * 100:.1f}%", fair_odds(pred.away_win))
        prob_table.add_section()

        # Goles
        prob_table.add_row("Totales (Mas/Menos 2.5)", "Over 2.5 Goles", f"{pred.over_2_5 * 100:.1f}%", fair_odds(pred.over_2_5))
        prob_table.add_row("", "Under 2.5 Goles", f"{pred.under_2_5 * 100:.1f}%", fair_odds(pred.under_2_5))
        prob_table.add_row("Totales (Mas/Menos 3.5)", "Over 3.5 Goles", f"{pred.over_3_5 * 100:.1f}%", fair_odds(pred.over_3_5))
        prob_table.add_row("", "Under 3.5 Goles", f"{pred.under_3_5 * 100:.1f}%", fair_odds(pred.under_3_5))
        prob_table.add_section()

        # BTTS
        prob_table.add_row("Ambos Equipos Marcan", "Si (BTTS Yes)", f"{pred.btts_yes * 100:.1f}%", fair_odds(pred.btts_yes))
        prob_table.add_row("", "No (BTTS No)", f"{pred.btts_no * 100:.1f}%", fair_odds(pred.btts_no))

        console.print(prob_table)

        # Goles esperados y marcadores más probables
        top_scores_str = ", ".join([f"{h}-{a} ({p * 100:.1f}%)" for h, a, p in pred.top_scores[:5]])
        console.print(f"[dim]xG Esperados:[/dim] [cyan]{player_a.display_name} {pred.expected_goals_home:.2f}[/cyan] - [cyan]{pred.expected_goals_away:.2f} {player_b.display_name}[/cyan] (Total: {pred.expected_total_goals:.2f})")
        console.print(f"[dim]Marcadores Top 5:[/dim] [bold]{top_scores_str}[/bold]\n")

        # 5. Evaluación de Apuestas de Valor (+EV) si se introducen cuotas
        if odds_1x2 or odds_o25 or odds_btts:
            eval_table = Table(title="[VALUE BETS] Analisis de Valor (+EV) & Desmargenado", show_header=True, header_style="bold yellow")
            eval_table.add_column("Mercado")
            eval_table.add_column("Seleccion")
            eval_table.add_column("Cuota Bookie", justify="center")
            eval_table.add_column("Prob. Implicita Real (Sin Margen)", justify="center")
            eval_table.add_column("Prob. Modelo", justify="center")
            eval_table.add_column("EV Edge %", justify="center")
            eval_table.add_column("Kelly Stake (1/4)", justify="center")

            if odds_1x2:
                vbs = evaluate_market_value(
                    Market.ONE_X_TWO,
                    {"home": odds_1x2[0], "draw": odds_1x2[1], "away": odds_1x2[2]},
                    {"home": pred.home_win, "draw": pred.draw, "away": pred.away_win},
                    min_edge=settings.MIN_EDGE_THRESHOLD,
                    kelly_fraction=settings.KELLY_FRACTION,
                    max_stake=settings.MAX_STAKE_PERCENT,
                )
                for vb in vbs:
                    color = "green" if vb.is_value else "white"
                    eval_table.add_row(
                        "1X2",
                        vb.selection,
                        f"@{vb.decimal_odds:.2f}",
                        f"{vb.fair_implied_prob * 100:.1f}%",
                        f"{vb.model_prob * 100:.1f}%",
                        f"[{color}]{vb.edge_percentage:+.1f}%[/{color}]",
                        f"[{color}]{vb.kelly_stake_pct * 100:.1f}%[/{color}]" if vb.is_value else "-"
                    )

            if odds_o25:
                vbs = evaluate_market_value(
                    Market.OVER_UNDER_2_5,
                    {"over": odds_o25[0], "under": odds_o25[1]},
                    {"over": pred.over_2_5, "under": pred.under_2_5},
                    min_edge=settings.MIN_EDGE_THRESHOLD,
                    kelly_fraction=settings.KELLY_FRACTION,
                    max_stake=settings.MAX_STAKE_PERCENT,
                )
                for vb in vbs:
                    color = "green" if vb.is_value else "white"
                    eval_table.add_row(
                        "O/U 2.5",
                        vb.selection,
                        f"@{vb.decimal_odds:.2f}",
                        f"{vb.fair_implied_prob * 100:.1f}%",
                        f"{vb.model_prob * 100:.1f}%",
                        f"[{color}]{vb.edge_percentage:+.1f}%[/{color}]",
                        f"[{color}]{vb.kelly_stake_pct * 100:.1f}%[/{color}]" if vb.is_value else "-"
                    )

            if odds_btts:
                vbs = evaluate_market_value(
                    Market.BTTS,
                    {"yes": odds_btts[0], "no": odds_btts[1]},
                    {"yes": pred.btts_yes, "no": pred.btts_no},
                    min_edge=settings.MIN_EDGE_THRESHOLD,
                    kelly_fraction=settings.KELLY_FRACTION,
                    max_stake=settings.MAX_STAKE_PERCENT,
                )
                for vb in vbs:
                    color = "green" if vb.is_value else "white"
                    eval_table.add_row(
                        "BTTS",
                        vb.selection,
                        f"@{vb.decimal_odds:.2f}",
                        f"{vb.fair_implied_prob * 100:.1f}%",
                        f"{vb.model_prob * 100:.1f}%",
                        f"[{color}]{vb.edge_percentage:+.1f}%[/{color}]",
                        f"[{color}]{vb.kelly_stake_pct * 100:.1f}%[/{color}]" if vb.is_value else "-"
                    )

            console.print(eval_table)
        else:
            console.print("[dim]Tip: Pasa cuotas con --odds-1x2 1.95 4.20 2.80 para ver calculo de Value Bets (+EV).[/dim]\n")

    finally:
        db.close()


def run_backtest_cli(
    format_str: str | None = None,
    bankroll: float = 1000.0,
    min_edge: float = 0.03,
    stake_mode: str = "kelly",
):
    """Ejecuta el backtesting histórico y reporta métricas completas."""
    fmt_map = {
        "4x2": DurationFormat.MIN_8,
        "8": DurationFormat.MIN_8,
        "5x2": DurationFormat.MIN_10,
        "10": DurationFormat.MIN_10,
        "6x2": DurationFormat.MIN_12,
        "12": DurationFormat.MIN_12,
    }
    duration_fmt = fmt_map.get(format_str.lower()) if format_str else None

    db = SessionLocal()
    try:
        console.print(f"\n[bold cyan]Iniciando simulacion historica Walk-Forward (Formato: {format_str or 'Todos'})...[/bold cyan]")
        engine = BacktestEngine(
            session=db,
            initial_bankroll=bankroll,
            min_edge_threshold=min_edge,
            stake_mode=stake_mode,
            kelly_fraction=settings.KELLY_FRACTION,
            max_stake_percent=settings.MAX_STAKE_PERCENT,
            decay_xi=settings.TIME_DECAY_XI,
        )

        report = engine.run(duration_format=duration_fmt, min_training_matches=15)

        # Tablas de reporte
        console.print()
        final_color = "green" if report.final_bankroll >= bankroll else "red"
        profit_color = "green" if report.total_profit >= 0 else "red"
        roi_color = "green" if report.roi >= 0 else "red"

        console.print(Panel.fit(
            f"[bold green]RESULTADOS DEL BACKTESTING HISTORICO[/bold green]\n"
            f"Partidos Evaluados: [bold]{report.total_matches_evaluated}[/] | Apuestas Colocadas: [bold]{report.total_bets_placed}[/]\n"
            f"Banca Inicial: [bold]${bankroll:.2f}[/] -> Final: [bold {final_color}]${report.final_bankroll:.2f}[/]\n"
            f"Beneficio Neto: [bold {profit_color}]${report.total_profit:+.2f}[/] | [bold {roi_color}]ROI: {report.roi:+.2f}%[/]",
            border_style="green"
        ))

        summary_table = Table(title="[RENDIMIENTO] Metricas Globales de Calidad Predictiva y Rendimiento", header_style="bold magenta")
        summary_table.add_column("Metrica")
        summary_table.add_column("Valor", justify="center")

        summary_table.add_row("Tasa de Acierto (Hit Rate)", f"{report.win_rate:.1f}% ({report.won_bets}/{report.total_bets_placed})")
        summary_table.add_row("Capital Total Apostado", f"${report.total_staked:.2f}")
        summary_table.add_row("Maximo Drawdown", f"{report.max_drawdown_pct:.1f}%")
        summary_table.add_row("Sharpe Ratio (Riesgo/Retorno)", f"{report.sharpe_ratio:.2f}")
        summary_table.add_row("Brier Score 1X2 (Menor = Mejor calibrado)", f"{report.global_brier_score_1x2:.4f}")
        summary_table.add_row("Brier Score Totales O/U 2.5", f"{report.global_brier_score_totals:.4f}")
        console.print(summary_table)

        if report.market_breakdown:
            mkt_table = Table(title="[MERCADOS] Desglose por Mercado", header_style="bold cyan")
            mkt_table.add_column("Mercado")
            mkt_table.add_column("Apuestas", justify="center")
            mkt_table.add_column("Win %", justify="center")
            mkt_table.add_column("Apostado", justify="center")
            mkt_table.add_column("Beneficio", justify="center")
            mkt_table.add_column("ROI %", justify="center")

            for m_name, perf in report.market_breakdown.items():
                pnl_color = "green" if perf.total_profit >= 0 else "red"
                mkt_table.add_row(
                    m_name,
                    str(perf.total_bets),
                    f"{perf.win_rate:.1f}%",
                    f"${perf.total_staked:.2f}",
                    f"[{pnl_color}]${perf.total_profit:+.2f}[/{pnl_color}]",
                    f"[{pnl_color}]{perf.roi:+.2f}%[/{pnl_color}]"
                )
            console.print(mkt_table)

        if report.format_breakdown and len(report.format_breakdown) > 1:
            fmt_table = Table(title="[FORMATOS] Desglose por Formato de Duracion", header_style="bold yellow")
            fmt_table.add_column("Formato")
            fmt_table.add_column("Apuestas", justify="center")
            fmt_table.add_column("Win %", justify="center")
            fmt_table.add_column("Apostado", justify="center")
            fmt_table.add_column("Beneficio", justify="center")
            fmt_table.add_column("ROI %", justify="center")

            for f_name, f_perf in report.format_breakdown.items():
                pnl_color = "green" if f_perf.total_profit >= 0 else "red"
                fmt_table.add_row(
                    f_name,
                    str(f_perf.total_bets),
                    f"{f_perf.win_rate:.1f}%",
                    f"${f_perf.total_staked:.2f}",
                    f"[{pnl_color}]${f_perf.total_profit:+.2f}[/{pnl_color}]",
                    f"[{pnl_color}]{f_perf.roi:+.2f}%[/{pnl_color}]"
                )
            console.print(fmt_table)

    finally:
        db.close()


def run_seed_cli(num_matches: int = 120):
    """Inicializa la base de datos con datos de prueba realistas."""
    init_db()
    db = SessionLocal()
    try:
        console.print(f"[bold cyan]Generando {num_matches} partidos por formato (4x2, 5x2, 6x2) con cuotas realistas...[/bold cyan]")
        count = seed_realistic_esoccer_dataset(db, num_matches_per_format=num_matches)
        console.print(f"[bold green][OK] Creados {count} partidos y snapshots de cuotas exitosamente en la base de datos.[/bold green]\n")
    finally:
        db.close()
