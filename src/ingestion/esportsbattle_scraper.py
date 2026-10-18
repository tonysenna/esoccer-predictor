from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import urllib3
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.analytics.elo import DynamicEloEngine
from src.db.models import DurationFormat, Market, Match, MatchStatus, OddsSnapshot, Player
from src.db.queries import get_or_create_player

urllib3.disable_warnings()

logger = logging.getLogger(__name__)

BASE_URL = "https://football.esportsbattle.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://football.esportsbattle.com/",
}

DATA_DIR = Path("data")
TEAMS_CACHE_FILE = DATA_DIR / "esportsbattle_teams.json"
PLAYERS_CACHE_FILE = DATA_DIR / "esportsbattle_players.json"

# Categorización de tiers para EA Sports FC / Esports Battle
TOP_TEAMS = {
    "Real Madrid", "Manchester City", "Bayern", "Bayern Munich", "PSG", "Paris Saint-Germain",
    "Liverpool", "Arsenal", "Barcelona", "Inter", "Inter Milan"
}
HIGH_TEAMS = {
    "Napoli", "Juventus", "Atletico Madrid", "Milan", "AC Milan", "Dortmund", "Borussia Dortmund",
    "Chelsea", "Newcastle", "Tottenham", "Bayer Leverkusen", "Aston Villa", "Porto", "Sporting",
    "Benfica", "Manchester United", "Roma"
}


def get_team_tier_modifier(team_name: str | None) -> float:
    """
    Calcula el ajuste en xG basado en la calidad del equipo (licencias oficiales de EA Sports FC).
    Top tier: +0.20 xG
    High tier: +0.08 xG
    Mid / Otros: 0.0 xG
    """
    if not team_name:
        return 0.0
    t = team_name.strip()
    if any(top.lower() == t.lower() for top in TOP_TEAMS):
        return 0.20
    if any(high.lower() == t.lower() for high in HIGH_TEAMS):
        return 0.08
    return 0.0


class EsportsBattleScraper:
    """
    Scraper y adaptador oficial para football.esportsbattle.com.
    Descarga equipos, participantes con estadísticas reales de carrera (win/loss/draw),
    partidos programados del día y resultados históricos.
    """

    def __init__(self, db: Session, timeout: int = 12):
        self.db = db
        self.timeout = timeout
        self.elo_engine = DynamicEloEngine()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any | None:
        url = f"{BASE_URL}{endpoint}" if endpoint.startswith("/") else f"{BASE_URL}/{endpoint}"
        try:
            resp = requests.get(url, headers=HEADERS, params=params, verify=False, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Error {resp.status_code} al consultar {url}")
        except Exception as e:
            logger.error(f"Excepción en petición a {url}: {e}")
        return None

    def sync_teams(self) -> list[str]:
        """
        Descarga y guarda los 246 equipos oficiales de Esports Battle.
        """
        data = self._get("/api/teams")
        team_names: list[str] = []
        if data and "teams" in data:
            for t in data["teams"]:
                name = t.get("token_international") or t.get("token")
                if name and name not in team_names:
                    team_names.append(name.strip())

        team_names.sort()
        if team_names:
            with open(TEAMS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(team_names, f, ensure_ascii=False, indent=2)
            logger.info(f"Sincronizados {len(team_names)} equipos de Esports Battle.")

        return team_names

    def get_all_teams(self) -> list[str]:
        """
        Devuelve la lista de equipos disponibles desde cache o API.
        Prioriza los equipos más comunes arriba.
        """
        if TEAMS_CACHE_FILE.exists():
            try:
                with open(TEAMS_CACHE_FILE, "r", encoding="utf-8") as f:
                    teams = json.load(f)
                    if teams:
                        return teams
            except Exception:
                pass
        return self.sync_teams()

    def sync_participants(self, max_pages: int = 17) -> int:
        """
        Sincroniza todos los jugadores reales de Esports Battle con sus
        estadísticas de carrera (partidos jugados, victorias, derrotas, empates, win rate).
        """
        synced_count = 0
        all_players_data = []

        for page in range(1, max_pages + 1):
            data = self._get("/api/participants", params={"page": page})
            if not data or "participants" not in data:
                break

            participants = data.get("participants", [])
            if not participants:
                break

            for p_info in participants:
                nickname = p_info.get("nickname")
                if not nickname:
                    continue

                nickname = nickname.strip()
                source_id = str(p_info.get("id", nickname))
                total_matches = int(p_info.get("totalMatches") or 0)
                total_win = int(p_info.get("totalWin") or 0)
                total_lose = int(p_info.get("totalLose") or 0)
                total_draw = int(p_info.get("totalDraw") or 0)

                # Si no vinieron los desgloses en la lista, consultar el perfil individual
                if total_matches > 0 and total_win == 0 and total_lose == 0:
                    detail = self._get(f"/api/participants/{nickname}")
                    if detail:
                        total_win = int(detail.get("totalWin") or 0)
                        total_lose = int(detail.get("totalLose") or 0)
                        total_draw = int(detail.get("totalDraw") or 0)

                win_rate = (total_win / total_matches) if total_matches > 0 else 0.0

                # Actualizar o crear en base de datos
                stmt = select(Player).where(
                    (Player.display_name == nickname) | (Player.source_id == source_id)
                )
                player = self.db.execute(stmt).scalar_one_or_none()
                if not player:
                    player = Player(
                        display_name=nickname,
                        source="esportsbattle",
                        source_id=source_id,
                        total_matches=total_matches,
                        total_win=total_win,
                        total_lose=total_lose,
                        total_draw=total_draw,
                        win_rate=win_rate,
                    )
                    self.db.add(player)
                else:
                    player.source = "esportsbattle"
                    player.total_matches = total_matches
                    player.total_win = total_win
                    player.total_lose = total_lose
                    player.total_draw = total_draw
                    player.win_rate = win_rate

                synced_count += 1
                all_players_data.append({
                    "nickname": nickname,
                    "id": source_id,
                    "total_matches": total_matches,
                    "total_win": total_win,
                    "total_lose": total_lose,
                    "total_draw": total_draw,
                    "win_rate": round(win_rate, 4),
                })

            if page >= data.get("totalPages", max_pages):
                break

        self.db.commit()

        if all_players_data:
            with open(PLAYERS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(all_players_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Sincronizados {synced_count} jugadores con sus estadísticas reales.")
        return synced_count

    def sync_upcoming_matches(self) -> list[dict[str, Any]]:
        """
        Descarga los partidos programados más cercanos desde /api/tournaments/nearest-matches
        y los indexa en la base de datos con equipos y jugadores.
        """
        raw_matches = self._get("/api/tournaments/nearest-matches")
        if not raw_matches or not isinstance(raw_matches, list):
            logger.warning("No se recibieron nearest-matches de Esports Battle.")
            return []

        indexed = []
        for m in raw_matches:
            p1 = m.get("participant1") or {}
            p2 = m.get("participant2") or {}
            nick1 = p1.get("nickname")
            nick2 = p2.get("nickname")
            if not nick1 or not nick2:
                continue

            t1_info = p1.get("team") or {}
            t2_info = p2.get("team") or {}
            team1 = t1_info.get("token_international") or t1_info.get("token") or "Equipo Local"
            team2 = t2_info.get("token_international") or t2_info.get("token") or "Equipo Visitante"

            m_id = str(m.get("id"))
            date_str = m.get("date")
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else datetime.now(timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)

            # Buscar o crear gamers
            player1 = get_or_create_player(self.db, display_name=nick1, source="esportsbattle")
            player2 = get_or_create_player(self.db, display_name=nick2, source="esportsbattle")

            # Buscar si el partido ya existe
            stmt = select(Match).where(Match.source == "esportsbattle", Match.source_match_id == m_id)
            match_obj = self.db.execute(stmt).scalar_one_or_none()

            tournament_name = "Esports Battle (8 min)"
            loc = m.get("location") or {}
            if loc.get("token_international"):
                tournament_name = f"Esports Battle - {loc.get('token_international')} (8 min)"

            if not match_obj:
                match_obj = Match(
                    source="esportsbattle",
                    source_match_id=m_id,
                    tournament=tournament_name,
                    duration_format=DurationFormat.MIN_8,
                    status=MatchStatus.SCHEDULED,
                    player_home_id=player1.id,
                    player_away_id=player2.id,
                    team_home=team1,
                    team_away=team2,
                    scheduled_at=dt,
                )
                self.db.add(match_obj)
                self.db.flush()

                # Añadir cuotas estimadas o de apertura por defecto si no existen
                for sel, val in [("home", 2.05), ("draw", 4.75), ("away", 2.40)]:
                    self.db.add(OddsSnapshot(
                        match_id=match_obj.id,
                        bookmaker="bet365",
                        market=Market.ONE_X_TWO,
                        selection=sel,
                        odds_value=val,
                        captured_at=dt,
                    ))
                for sel, val in [("over", 1.10), ("under", 6.25)]:
                    self.db.add(OddsSnapshot(
                        match_id=match_obj.id,
                        bookmaker="bet365",
                        market=Market.OVER_UNDER_2_5,
                        selection=sel,
                        odds_value=val,
                        captured_at=dt,
                    ))
            else:
                match_obj.team_home = team1
                match_obj.team_away = team2
                match_obj.scheduled_at = dt

            indexed.append({
                "id": match_obj.id,
                "source_id": m_id,
                "team_home": team1,
                "gamer_home": nick1,
                "team_away": team2,
                "gamer_away": nick2,
                "scheduled_at": dt,
                "tournament": tournament_name,
            })

        self.db.commit()
        logger.info(f"Indexados {len(indexed)} partidos programados de Esports Battle.")
        return indexed

    def sync_daily_tournaments_matches(self, max_players: int = 5) -> int:
        """
        Recorre torneos recientes de los jugadores activos para extraer todos los partidos
        del día (tanto programados con equipos como finalizados con goles).
        """
        data = self._get("/api/participants", params={"page": 1})
        if not data or "participants" not in data:
            return 0

        participants = data.get("participants", [])[:max_players]
        seen_tournaments = set()
        total_synced_matches = 0

        for p in participants:
            nick = p.get("nickname")
            if not nick:
                continue

            t_data = self._get(f"/api/participants/{nick}/tournaments")
            if not t_data or "tournaments" not in t_data:
                continue

            for t_item in t_data.get("tournaments", [])[:4]:
                t_id = t_item.get("id")
                if not t_id or t_id in seen_tournaments:
                    continue
                seen_tournaments.add(t_id)

                matches_data = self._get(f"/api/tournaments/{t_id}/matches")
                if not matches_data or not isinstance(matches_data, list):
                    continue

                for m in matches_data:
                    p1 = m.get("participant1") or {}
                    p2 = m.get("participant2") or {}
                    n1 = p1.get("nickname")
                    n2 = p2.get("nickname")
                    if not n1 or not n2:
                        continue

                    t1_info = p1.get("team") or {}
                    t2_info = p2.get("team") or {}
                    team1 = t1_info.get("token_international") or t1_info.get("token") or "Equipo Local"
                    team2 = t2_info.get("token_international") or t2_info.get("token") or "Equipo Visitante"

                    m_id = str(m.get("id"))
                    date_str = m.get("date")
                    try:
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else datetime.now(timezone.utc)
                    except Exception:
                        dt = datetime.now(timezone.utc)

                    player1 = get_or_create_player(self.db, display_name=n1, source="esportsbattle")
                    player2 = get_or_create_player(self.db, display_name=n2, source="esportsbattle")

                    status_id = m.get("status_id")
                    # status_id == 3 significa terminado con marcador
                    is_finished = (status_id == 3)
                    score1 = p1.get("score")
                    score2 = p2.get("score")

                    stmt = select(Match).where(Match.source == "esportsbattle", Match.source_match_id == m_id)
                    match_obj = self.db.execute(stmt).scalar_one_or_none()

                    tourn_title = t_item.get("token_international") or "Esports Battle (8 min)"

                    if not match_obj:
                        match_obj = Match(
                            source="esportsbattle",
                            source_match_id=m_id,
                            tournament=tourn_title,
                            duration_format=DurationFormat.MIN_8,
                            status=MatchStatus.FINISHED if is_finished else MatchStatus.SCHEDULED,
                            player_home_id=player1.id,
                            player_away_id=player2.id,
                            team_home=team1,
                            team_away=team2,
                            goals_home=int(score1) if (is_finished and score1 is not None) else None,
                            goals_away=int(score2) if (is_finished and score2 is not None) else None,
                            scheduled_at=dt,
                            finished_at=dt if is_finished else None,
                        )
                        self.db.add(match_obj)
                        self.db.flush()

                        if is_finished and match_obj.goals_home is not None and match_obj.goals_away is not None:
                            self.elo_engine.update_ratings_for_match(self.db, match_obj, commit=False)

                        total_synced_matches += 1
                    else:
                        match_obj.team_home = team1
                        match_obj.team_away = team2
                        if is_finished and score1 is not None and score2 is not None:
                            match_obj.status = MatchStatus.FINISHED
                            match_obj.goals_home = int(score1)
                            match_obj.goals_away = int(score2)
                            match_obj.finished_at = dt

        self.db.commit()
        return total_synced_matches
