import argparse
import sys
from src.db.session import init_db
from src.ui.cli import run_backtest_cli, run_predict_cli, run_seed_cli


def main():
    parser = argparse.ArgumentParser(description="e-Soccer / FIFA Predictive Engine & Value Bet Analyzer")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # Command: init
    init_parser = subparsers.add_parser("init", help="Inicializar base de datos SQLite")

    # Command: seed
    seed_parser = subparsers.add_parser("seed", help="Poblar base de datos con partidos y cuotas sintéticas realistas")
    seed_parser.add_argument("--matches", type=int, default=120, help="Número de partidos por formato (default: 120)")

    # Command: predict
    predict_parser = subparsers.add_parser("predict", help="Predecir partido entre Gamer A y Gamer B")
    predict_parser.add_argument("gamer_a", type=str, help="Nombre del Gamer Local")
    predict_parser.add_argument("gamer_b", type=str, help="Nombre del Gamer Visitante")
    predict_parser.add_argument("--format", type=str, default="5x2", choices=["4x2", "5x2", "6x2", "8", "10", "12"], help="Formato de duración (default: 5x2)")
    predict_parser.add_argument("--odds-1x2", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"), help="Cuotas 1X2 de la casa (ej: --odds-1x2 2.05 4.10 2.80)")
    predict_parser.add_argument("--odds-o25", nargs=2, type=float, metavar=("OVER", "UNDER"), help="Cuotas Más/Menos 2.5 (ej: --odds-o25 1.70 2.10)")
    predict_parser.add_argument("--odds-btts", nargs=2, type=float, metavar=("YES", "NO"), help="Cuotas Ambos Marcan (ej: --odds-btts 1.55 2.30)")

    # Command: backtest
    backtest_parser = subparsers.add_parser("backtest", help="Ejecutar simulación histórica Walk-Forward")
    backtest_parser.add_argument("--format", type=str, default=None, choices=["4x2", "5x2", "6x2", "8", "10", "12"], help="Filtrar por formato o dejar vacío para todos")
    backtest_parser.add_argument("--bankroll", type=float, default=1000.0, help="Banca inicial en $ (default: 1000.0)")
    backtest_parser.add_argument("--min-edge", type=float, default=0.03, help="Edge mínimo +EV (default: 0.03 = 3%%)")
    backtest_parser.add_argument("--stake-mode", type=str, default="kelly", choices=["kelly", "flat"], help="Modo de gestión de banca (default: kelly)")

    # Command: import-csv
    csv_parser = subparsers.add_parser("import-csv", help="Importar partidos reales desde archivo CSV")
    csv_parser.add_argument("file", type=str, help="Ruta al archivo CSV con partidos y cuotas")

    # Command: sync-api
    api_parser = subparsers.add_parser("sync-api", help="Sincronizar partidos reales desde BetsAPI")
    api_parser.add_argument("--token", type=str, required=True, help="Token API de BetsAPI")
    api_parser.add_argument("--days", type=int, default=30, help="Días hacia atrás a sincronizar")

    args = parser.parse_args()

    if args.command == "init":
        init_db()
        print("Base de datos inicializada exitosamente.")
    elif args.command == "seed":
        run_seed_cli(num_matches=args.matches)
    elif args.command == "predict":
        run_predict_cli(
            gamer_a_name=args.gamer_a,
            gamer_b_name=args.gamer_b,
            format_str=args.format,
            odds_1x2=tuple(args.odds_1x2) if args.odds_1x2 else None,
            odds_o25=tuple(args.odds_o25) if args.odds_o25 else None,
            odds_btts=tuple(args.odds_btts) if args.odds_btts else None,
        )
    elif args.command == "backtest":
        run_backtest_cli(
            format_str=args.format,
            bankroll=args.bankroll,
            min_edge=args.min_edge,
            stake_mode=args.stake_mode,
        )
    elif args.command == "import-csv":
        from src.db.session import SessionLocal
        from src.ingestion.csv_importer import import_matches_from_csv
        with SessionLocal() as db:
            cnt = import_matches_from_csv(db, args.file)
            print(f"Importados {cnt} partidos reales desde {args.file} correctamente.")
    elif args.command == "sync-api":
        from src.db.session import SessionLocal
        from src.ingestion.betsapi_adapter import BetsAPIAdapter
        with SessionLocal() as db:
            adapter = BetsAPIAdapter(api_token=args.token)
            print(f"Sincronizando partidos reales de BetsAPI (ultimos {args.days} dias)...")
            from datetime import datetime, timedelta, timezone
            end_d = datetime.now(timezone.utc)
            start_d = end_d - timedelta(days=args.days)
            raw_past = adapter.fetch_past_results(start_d, end_d)
            past_m, past_o = adapter.sync_to_db(db, raw_past, fetch_odds=False)
            print(f"✅ Descargados e indexados {past_m} partidos historicos reales.")

            print("Sincronizando cartelera de hoy y proximos partidos en vivo...")
            raw_upcoming = adapter.fetch_upcoming_matches()
            up_m, up_o = adapter.sync_to_db(db, raw_upcoming, fetch_odds=True)
            print(f"✅ Indexados {up_m} partidos futuros con sus cuotas de Bet365.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
