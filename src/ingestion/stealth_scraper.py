from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random
import re
import time
from typing import Any
from bs4 import BeautifulSoup
import requests
from sqlalchemy.orm import Session

from src.analytics.elo import DynamicEloEngine
from src.db.models import DurationFormat, Market, Match, MatchStatus, OddsSnapshot, Player
from src.db.queries import get_or_create_player
from src.ingestion.validators import validate_match_payload


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
]


@dataclass
class ScrapedMatchResult:
    source_id: str
    tournament_name: str
    duration_format: DurationFormat
    player_home: str
    player_away: str
    team_home: str | None = None
    team_away: str | None = None
    goals_home: int | None = None
    goals_away: int | None = None
    match_time: datetime = datetime.now(timezone.utc)
    status: MatchStatus = MatchStatus.FINISHED
    odds_1x2: tuple[float, float, float] | None = None
    odds_o25: tuple[float, float] | None = None
    odds_btts: tuple[float, float] | None = None


class StealthSession:
    """
    Sesión HTTP con evasión avanzada de bloqueos:
    - Rotación de huellas de navegador y headers Client-Hints.
    - Delays con distribución gaussiana (Jitter humano).
    - Reintentos con retroceso exponencial adaptativo y evasión de trampas.
    """

    def __init__(self, min_delay: float = 1.2, max_delay: float = 2.8):
        self.session = requests.Session()
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._update_stealth_headers()

    def _update_stealth_headers(self, referer: str | None = None):
        ua = random.choice(USER_AGENTS)
        is_windows = "Windows" in ua
        platform = '"Windows"' if is_windows else '"macOS"'

        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none" if not referer else "same-origin",
            "Sec-Fetch-User": "?1",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": platform,
        }
        if referer:
            headers["Referer"] = referer
        self.session.headers.update(headers)

    def get(self, url: str, max_retries: int = 4, referer: str | None = None) -> requests.Response | None:
        """Petición GET protegida con jitter y reintentos adaptativos."""
        self._update_stealth_headers(referer=referer)

        for attempt in range(1, max_retries + 1):
            try:
                # Simular tiempo de lectura/navegación humana con jitter gaussiano
                jitter = random.gauss(self.min_delay, 0.4)
                sleep_time = max(0.8, min(self.max_delay + 2.0, jitter))
                time.sleep(sleep_time)

                resp = self.session.get(url, timeout=12)

                # Detección de Rate Limit (429) o Cloudflare WAF (403/503)
                if resp.status_code in [429, 403, 503]:
                    backoff = (2 ** attempt) + random.uniform(1.0, 3.0)
                    time.sleep(backoff)
                    self._update_stealth_headers()
                    continue

                if resp.status_code == 200:
                    return resp

            except (requests.RequestException, TimeoutError):
                time.sleep(1.5 * attempt)

        return None


class IntelligentEsoccerScraper:
    """
    Extractor inteligente de partidos de e-Soccer (GT Leagues y Esports Battle).
    Parsea formatos de duración, limpia nombres de gamers y persiste con ratings ELO automáticos.
    """

    def __init__(self, session_db: Session):
        self.db = session_db
        self.http = StealthSession()
        self.elo_engine = DynamicEloEngine()

    def parse_format_from_text(self, text: str) -> DurationFormat:
        """Detecta con heurística robusta el formato 4x2 (8m), 5x2 (10m) o 6x2 (12m)."""
        t = text.lower()
        if "8 min" in t or "4x2" in t or "8m" in t or "8 mins" in t:
            return DurationFormat.MIN_8
        elif "12 min" in t or "6x2" in t or "12m" in t or "12 mins" in t:
            return DurationFormat.MIN_12
        else:
            return DurationFormat.MIN_10  # 5x2 (10 min) es el formato estándar predominante

    def clean_gamer_name(self, raw_name: str) -> str:
        """Limpia nombres eliminando etiquetas de equipos o prefijos entre paréntesis."""
        cleaned = re.sub(r"\(.*?\)", "", raw_name)
        cleaned = re.sub(r"\[.*?\]", "", cleaned)
        cleaned = re.sub(r"[^\w\s\d_-]", "", cleaned)
        return cleaned.strip()

    def ingest_scraped_match(self, match_data: ScrapedMatchResult) -> bool:
        """Guarda el partido en la base de datos y actualiza los ratings ELO."""
        # Verificar duplicados por source_match_id
        existing = self.db.query(Match).filter_by(source_match_id=match_data.source_id).first()
        if existing:
            return False

        # Obtener o registrar jugadores
        p_home = get_or_create_player(self.db, display_name=match_data.player_home, source="scraped")
        p_away = get_or_create_player(self.db, display_name=match_data.player_away, source="scraped")

        if p_home.id == p_away.id:
            return False

        is_finished = match_data.status == MatchStatus.FINISHED
        match_obj = Match(
            source="scraped",
            source_match_id=match_data.source_id,
            tournament=match_data.tournament_name,
            duration_format=match_data.duration_format,
            status=match_data.status,
            player_home_id=p_home.id,
            player_away_id=p_away.id,
            team_home=match_data.team_home,
            team_away=match_data.team_away,
            goals_home=match_data.goals_home if is_finished else None,
            goals_away=match_data.goals_away if is_finished else None,
            scheduled_at=match_data.match_time,
            finished_at=match_data.match_time + timedelta(minutes=15) if is_finished else None,
        )
        self.db.add(match_obj)
        self.db.flush()

        # Guardar cuotas
        if match_data.odds_1x2:
            o_h, o_d, o_a = match_data.odds_1x2
            for sel, val in [("home", o_h), ("draw", o_d), ("away", o_a)]:
                self.db.add(
                    OddsSnapshot(
                        match_id=match_obj.id,
                        bookmaker="bet365",
                        market=Market.ONE_X_TWO,
                        selection=sel,
                        odds_value=val,
                        captured_at=match_data.match_time,
                        is_closing_line=is_finished,
                    )
                )

        if match_data.odds_o25:
            o_over, o_under = match_data.odds_o25
            self.db.add(OddsSnapshot(match_id=match_obj.id, bookmaker="bet365", market=Market.OVER_UNDER_2_5, selection="over", odds_value=o_over, captured_at=match_data.match_time, is_closing_line=is_finished))
            self.db.add(OddsSnapshot(match_id=match_obj.id, bookmaker="bet365", market=Market.OVER_UNDER_2_5, selection="under", odds_value=o_under, captured_at=match_data.match_time, is_closing_line=is_finished))

        if match_data.odds_btts:
            o_byes, o_bno = match_data.odds_btts
            self.db.add(OddsSnapshot(match_id=match_obj.id, bookmaker="bet365", market=Market.BTTS, selection="yes", odds_value=o_byes, captured_at=match_data.match_time, is_closing_line=is_finished))
            self.db.add(OddsSnapshot(match_id=match_obj.id, bookmaker="bet365", market=Market.BTTS, selection="no", odds_value=o_bno, captured_at=match_data.match_time, is_closing_line=is_finished))

        # Actualizar rating ELO si el partido está finalizado
        if is_finished:
            self.elo_engine.update_ratings_for_match(self.db, match_obj, commit=False)
        self.db.commit()
        return True

    def extract_from_raw_feed(self, feed_items: list[dict[str, Any]]) -> int:
        """
        Ingiere lotes de partidos procesados de feeds públicos/JSON normalizados.
        """
        imported_count = 0
        for item in feed_items:
            home_name = self.clean_gamer_name(item.get("home", "Gamer1"))
            away_name = self.clean_gamer_name(item.get("away", "Gamer2"))
            tourn = item.get("tournament", "GT Leagues 10 min")
            fmt = self.parse_format_from_text(tourn)
            
            m_time = item.get("time") or datetime.now(timezone.utc)
            if isinstance(m_time, str):
                try:
                    m_time = datetime.fromisoformat(m_time)
                except ValueError:
                    m_time = datetime.now(timezone.utc)

            uid = item.get("id") or f"ext_{home_name}_{away_name}_{int(m_time.timestamp())}"

            m_status = item.get("status", MatchStatus.FINISHED)
            gh_val = item.get("goals_home")
            ga_val = item.get("goals_away")
            gh = int(gh_val) if (m_status == MatchStatus.FINISHED and gh_val is not None) else None
            ga = int(ga_val) if (m_status == MatchStatus.FINISHED and ga_val is not None) else None

            res = ScrapedMatchResult(
                source_id=str(uid),
                tournament_name=tourn,
                duration_format=fmt,
                player_home=home_name,
                player_away=away_name,
                goals_home=gh,
                goals_away=ga,
                match_time=m_time,
                status=m_status,
                odds_1x2=item.get("odds_1x2"),
                odds_o25=item.get("odds_o25"),
                odds_btts=item.get("odds_btts"),
            )
            if self.ingest_scraped_match(res):
                imported_count += 1

        return imported_count
