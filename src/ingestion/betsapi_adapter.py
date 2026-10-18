from __future__ import annotations

from datetime import datetime, timezone, timedelta
import time
from typing import Any
import httpx
from loguru import logger
from sqlalchemy.orm import Session

from src.db.models import DurationFormat, Market, Match, MatchStatus, OddsSnapshot
from src.db.queries import get_or_create_player
from src.ingestion.base import BaseIngestionAdapter


class BetsAPIAdapter(BaseIngestionAdapter):
    """
    Adaptador para ingesta de datos en vivo y resultados históricos desde BetsAPI (b365api.com).
    Filtra automáticamente ligas de e-Soccer FIFA (GT Leagues, Esports Battle, Adriatic, etc.).
    """

    ESPORT_KEYWORDS = ["esoccer", "gt leagues", "battle", "adriatic", "fifa", "esports"]

    def __init__(self, api_token: str | None = None, base_url: str = "https://api.b365api.com/v1"):
        self.api_token = api_token
        self.base_url = base_url
        self.client = httpx.Client(timeout=15.0)
        self.last_request_time = 0.0
        self.min_interval = 1.0  # Rate limit: 1 seg entre peticiones

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def _is_esoccer(self, league_name: str) -> bool:
        ln = league_name.lower()
        return any(k in ln for k in self.ESPORT_KEYWORDS)

    def _clean_gamer_name(self, raw_name: str) -> str:
        """Limpia nombres tipo 'Napoli (Dicca)' -> 'Dicca' o 'Dicca'."""
        if "(" in raw_name and ")" in raw_name:
            import re
            m = re.search(r"\(([^)]+)\)", raw_name)
            if m:
                return m.group(1).strip()
        return raw_name.strip()

    def fetch_upcoming_matches(self) -> list[dict[str, Any]]:
        """Consulta partidos programados en vivo de e-Soccer FIFA en Bet365."""
        if not self.api_token:
            logger.warning("BETSAPI_KEY no configurada. Omitiendo fetch.")
            return []

        self._rate_limit()
        try:
            url = f"{self.base_url}/events/upcoming"
            params = {"token": self.api_token, "sport_id": "1"}
            res = self.client.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            all_events = data.get("results", [])
            esoccer_events = [
                ev for ev in all_events
                if self._is_esoccer(ev.get("league", {}).get("name", ""))
            ]
            logger.info(f"BetsAPI: Encontrados {len(esoccer_events)} partidos futuros de e-Soccer.")
            return esoccer_events
        except Exception as e:
            logger.error(f"Error al obtener partidos futuros de BetsAPI: {e}")
            return []

    def fetch_past_results(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """Consulta resultados históricos finalizados de e-Soccer."""
        if not self.api_token:
            return []

        results = []
        cur_date = start_date
        while cur_date <= end_date:
            day_str = cur_date.strftime("%Y%m%d")
            self._rate_limit()
            try:
                url = f"{self.base_url}/events/ended"
                params = {"token": self.api_token, "sport_id": "1", "day": day_str}
                res = self.client.get(url, params=params)
                res.raise_for_status()
                data = res.json()
                day_events = data.get("results", [])
                esoccer_day = [
                    ev for ev in day_events
                    if self._is_esoccer(ev.get("league", {}).get("name", ""))
                ]
                results.extend(esoccer_day)
                logger.info(f"BetsAPI [{day_str}]: Descargados {len(esoccer_day)} partidos de e-Soccer.")
            except Exception as e:
                logger.error(f"Error en fecha {day_str} de BetsAPI: {e}")

            cur_date += timedelta(days=1)

        return results

    def fetch_odds_for_event(self, event_id: str) -> dict[str, Any] | None:
        """Obtiene cuotas de Bet365 para un evento específico."""
        if not self.api_token:
            return None

        self._rate_limit()
        try:
            url = "https://api.b365api.com/v2/event/odds"
            params = {"token": self.api_token, "event_id": event_id}
            res = self.client.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            return data.get("results", {})
        except Exception as e:
            logger.debug(f"No se pudieron obtener cuotas para evento {event_id}: {e}")
            return None

    def sync_to_db(self, session: Session, raw_data: list[dict[str, Any]], fetch_odds: bool = True) -> tuple[int, int]:
        """Convierte los payloads JSON de la API a modelos de base de datos SQLAlchemy."""
        new_matches = 0
        new_odds = 0

        for item in raw_data:
            try:
                m_id = str(item.get("id"))
                if not m_id:
                    continue

                h_raw = item.get("home", {}).get("name", "Gamer A")
                a_raw = item.get("away", {}).get("name", "Gamer B")
                h_name = self._clean_gamer_name(h_raw)
                a_name = self._clean_gamer_name(a_raw)

                p_home = get_or_create_player(session, display_name=h_name, source="betsapi")
                p_away = get_or_create_player(session, display_name=a_name, source="betsapi")

                t_name = item.get("league", {}).get("name", "GT Leagues 10 min")
                if "8" in t_name or "4x2" in t_name:
                    fmt = DurationFormat.MIN_8
                elif "12" in t_name or "6x2" in t_name:
                    fmt = DurationFormat.MIN_12
                else:
                    fmt = DurationFormat.MIN_10

                scheduled_ts = item.get("time", int(time.time()))
                try:
                    scheduled_dt = datetime.fromtimestamp(int(scheduled_ts), tz=timezone.utc)
                except Exception:
                    scheduled_dt = datetime.now(timezone.utc)

                match = session.query(Match).filter_by(source_match_id=m_id).first()
                if not match:
                    match = Match(
                        source="betsapi",
                        source_match_id=m_id,
                        tournament=t_name,
                        duration_format=fmt,
                        status=MatchStatus.SCHEDULED,
                        player_home_id=p_home.id,
                        player_away_id=p_away.id,
                        scheduled_at=scheduled_dt,
                    )
                    session.add(match)
                    session.flush()
                    new_matches += 1

                # Parsear marcador si finalizó
                ss = item.get("ss")
                if ss and "-" in ss:
                    parts = ss.split("-")
                    try:
                        match.goals_home = int(parts[0])
                        match.goals_away = int(parts[1])
                        match.status = MatchStatus.FINISHED
                        match.finished_at = scheduled_dt
                    except ValueError:
                        pass

                # Guardar cuotas si están disponibles
                if fetch_odds and match.id:
                    odds_data = self.fetch_odds_for_event(m_id)
                    if odds_data:
                        b365_odds = odds_data.get("odds", {}).get("1_1", [])
                        if b365_odds:
                            # 1X2 Full Time
                            for entry in b365_odds:
                                sel = entry.get("name", "").lower()
                                odd_val = float(entry.get("odds", 0.0))
                                if odd_val > 1.0:
                                    s_key = "home" if "1" in sel or sel == "home" else ("draw" if "x" in sel or sel == "draw" else "away")
                                    session.add(OddsSnapshot(
                                        match_id=match.id,
                                        bookmaker="bet365",
                                        market=Market.ONE_X_TWO,
                                        selection=s_key,
                                        odds_value=odd_val,
                                        captured_at=datetime.now(timezone.utc),
                                        is_closing_line=(match.status == MatchStatus.FINISHED),
                                    ))
                                    new_odds += 1

            except Exception as ex:
                logger.error(f"Error procesando evento de BetsAPI: {ex}")

        session.commit()
        return new_matches, new_odds
