from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from sqlalchemy.orm import Session

from src.db.models import DurationFormat, MatchStatus
from src.ingestion.stealth_scraper import IntelligentEsoccerScraper, ScrapedMatchResult


def import_matches_from_csv(session_db: Session, csv_path: str) -> int:
    """
    Importa partidos reales desde un archivo CSV con columnas:
    home, away, goals_home, goals_away, tournament/format, date/time, odds_1, odds_x, odds_2, odds_over25, odds_under25
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")

    scraper = IntelligentEsoccerScraper(session_db)
    imported = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            home = row.get("home", "").strip()
            away = row.get("away", "").strip()
            if not home or not away:
                continue

            gh_str = row.get("goals_home") or row.get("gh")
            ga_str = row.get("goals_away") or row.get("ga")
            
            is_finished = gh_str is not None and ga_str is not None and gh_str != "" and ga_str != ""
            gh = int(gh_str) if is_finished else None
            ga = int(ga_str) if is_finished else None

            tourn = row.get("tournament") or row.get("format") or "GT Leagues 10 min"
            fmt = scraper.parse_format_from_text(tourn)

            date_str = row.get("date") or row.get("time") or row.get("timestamp")
            m_time = datetime.now(timezone.utc)
            if date_str:
                for fmt_try in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M", "%d/%m/%Y"]:
                    try:
                        m_time = datetime.strptime(date_str.strip(), fmt_try).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        pass

            uid = row.get("id") or f"csv_{home}_{away}_{int(m_time.timestamp())}"

            # Cuotas
            o_h = float(row["odds_1"]) if "odds_1" in row and row["odds_1"] else None
            o_x = float(row["odds_x"]) if "odds_x" in row and row["odds_x"] else None
            o_a = float(row["odds_2"]) if "odds_2" in row and row["odds_2"] else None
            odds_1x2 = (o_h, o_x, o_a) if (o_h and o_x and o_a) else None

            o_o25 = float(row["odds_over25"]) if "odds_over25" in row and row["odds_over25"] else None
            o_u25 = float(row["odds_under25"]) if "odds_under25" in row and row["odds_under25"] else None
            odds_o25 = (o_o25, o_u25) if (o_o25 and o_u25) else None

            res = ScrapedMatchResult(
                source_id=str(uid),
                tournament_name=tourn,
                duration_format=fmt,
                player_home=home,
                player_away=away,
                goals_home=gh,
                goals_away=ga,
                match_time=m_time,
                status=MatchStatus.FINISHED if is_finished else MatchStatus.SCHEDULED,
                odds_1x2=odds_1x2,
                odds_o25=odds_o25,
            )

            if scraper.ingest_scraped_match(res):
                imported += 1

    return imported
