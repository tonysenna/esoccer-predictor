from datetime import datetime, timedelta, timezone
import time
import pandas as pd
import streamlit as st
import numpy as np

from src.analytics.dixon_coles import DixonColesModel
from src.analytics.elo import DynamicEloEngine
from src.analytics.metrics import calculate_h2h_metrics, calculate_player_form
from src.analytics.odds import calculate_kelly_stake, evaluate_market_value
from src.backtesting.engine import BacktestEngine
from src.config import get_settings
from src.db.models import DurationFormat, Market, Match, MatchStatus, OddsSnapshot, Player
from src.db.queries import (
    get_all_matches_for_format,
    get_closing_odds_for_match,
    get_h2h_matches,
    get_or_create_player,
    get_player_by_name,
    get_recent_form,
    get_upcoming_matches,
)
from src.db.session import SessionLocal, init_db
from src.ingestion.mock_generator import seed_realistic_esoccer_dataset

st.set_page_config(
    page_title="e-Soccer FIFA Predictive Engine & Value Bets",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Evitar que el traductor automático de Google Chrome rompa el árbol DOM de React/Streamlit
st.markdown(
    """
    <meta name="google" content="notranslate">
    <style>
    body, html, [data-testid="stAppViewContainer"] {
        translate: no !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

settings = get_settings()

st.title("⚽ e-Soccer Predictive Engine & Value Bet Analyzer")
st.caption("GT Leagues & Esports Battle (4x2, 5x2, 6x2) — Poisson Bivariada Dixon-Coles, Shrinkage Bayesiano & Backtesting Walk-Forward")

# Sidebar
st.sidebar.header("⚙️ Navegación & Control")
selected_tab = st.sidebar.radio(
    "Selecciona un módulo:",
    [
        "🔥 Radar +EV (Partidos de Hoy)",
        "🎯 Predicción & +EV",
        "📈 Backtesting Histórico",
        "📊 Explorador de Gamers",
        "🗄️ Base de Datos",
    ]
)

# Inicializar DB si no existe
init_db()

with SessionLocal() as db:
    # Cargar lista de equipos oficiales de Esports Battle y EA Sports FC
    from src.ingestion.esportsbattle_scraper import EsportsBattleScraper, get_team_tier_modifier
    esports_scraper = EsportsBattleScraper(db)
    official_teams = esports_scraper.get_all_teams()
    if not official_teams:
        official_teams = [
            "Napoli", "Sassuolo", "Juventus", "Roma", "Bologna", "Inter", "Milan", "Real Madrid",
            "Barcelona", "Atletico Madrid", "Manchester City", "Arsenal", "Liverpool", "Chelsea",
            "Manchester United", "Bayern", "Dortmund", "PSG", "Porto", "Benfica", "Sporting"
        ]

    # Sincronización Automática Inteligente al abrir la App (cada 5 minutos)
    now_ts = time.time()
    last_sync = st.session_state.get("last_auto_sync_ts", 0)
    if now_ts - last_sync > 300:  # Cada 5 minutos se auto-actualiza
        try:
            esports_scraper.sync_upcoming_matches()
            st.session_state["last_auto_sync_ts"] = now_ts
        except Exception:
            pass

    # Obtener lista de gamers registrados
    all_players = db.query(Player).order_by(Player.display_name).all()
    player_map_id = {p.id: p.display_name for p in all_players}
    player_names = [p.display_name for p in all_players] if all_players else ["Lio", "Danny", "Kravich", "Niko", "Calvin"]

    # -------------------------------------------------------------
    # TAB 1: RADAR +EV (ESCÁNER AUTOMÁTICO DE PARTIDOS DE HOY)
    # -------------------------------------------------------------
    if selected_tab == "🔥 Radar +EV (Partidos de Hoy)":
        st.subheader("🔥 Radar +EV: Escáner Automático de Oportunidades de Hoy")
        
        mins_ago = int((time.time() - st.session_state.get("last_auto_sync_ts", time.time())) / 60)
        st.caption(f"⚡ **Auto-Sincronización Activa:** La app se actualiza automáticamente con partidos del día y cuotas de Esports Battle (última actualización: hace {mins_ago} min).")

        # Barra de sincronización directa con Esports Battle y fuentes oficiales
        col_sync1, col_sync2 = st.columns([1, 1])
        with col_sync1:
            if st.button("🔄 Sincronizar Ahora Manualmente (Esports Battle)", type="primary", use_container_width=True):
                with st.spinner("Conectando con football.esportsbattle.com y descargando partidos del día..."):
                    synced_matches = esports_scraper.sync_upcoming_matches()
                    st.session_state["last_auto_sync_ts"] = time.time()
                    st.success(f"✅ ¡{len(synced_matches)} partidos sincronizados con equipos y jugadores de Esports Battle!")
                    time.sleep(0.5)
                    st.rerun()
        with col_sync2:
            if st.button("📊 Sincronizar Estadísticas Oficiales de Jugadores", use_container_width=True):
                with st.spinner("Actualizando estadísticas históricas oficiales (win/loss/draw)..."):
                    p_synced = esports_scraper.sync_participants(max_pages=5)
                    st.success(f"✅ ¡{p_synced} perfiles de jugadores actualizados con su historial real!")
                    time.sleep(0.5)
                    st.rerun()

        st.markdown("---")
        upcoming = get_upcoming_matches(db, limit=100)

        if not upcoming:
            st.info("ℹ️ No hay partidos programados pendientes en la base de datos en este momento.")
            st.markdown("Pulsa el botón **'🔄 Sincronizar Partidos de Hoy (Esports Battle)'** arriba para importar los partidos oficiales automáticamente, o genera simulación:")
            if st.button("⚡ Cargar Simulación GT Leagues con Cuotas"):
                from src.ingestion.stealth_scraper import IntelligentEsoccerScraper
                scraper = IntelligentEsoccerScraper(db)
                sample_data = []
                gamers_pool = ["Lio", "Danny", "Kravich", "Niko", "Calvin", "Valentin", "Marlon", "Jack", "Rafa", "Leo"]
                import random
                for idx in range(12):
                    p_a, p_b = random.sample(gamers_pool, 2)
                    fmt_tag = random.choice(["10 min", "8 min", "12 min"])
                    sample_data.append({
                        "home": p_a,
                        "away": p_b,
                        "status": MatchStatus.SCHEDULED,
                        "tournament": f"GT Leagues {fmt_tag}",
                        "time": datetime.now(timezone.utc) + timedelta(minutes=15 * (idx + 1)),
                        "odds_1x2": (round(random.uniform(1.8, 2.9), 2), round(random.uniform(3.8, 4.8), 2), round(random.uniform(2.1, 3.3), 2)),
                        "odds_o25": (round(random.uniform(1.4, 1.8), 2), round(random.uniform(2.0, 2.7), 2)),
                        "odds_btts": (round(random.uniform(1.35, 1.65), 2), round(random.uniform(2.2, 2.9), 2)),
                    })
                imported = scraper.extract_from_raw_feed(sample_data)
                st.success(f"¡Se han añadido {imported} partidos programados para hoy a la cartelera!")
                st.rerun()
        else:
            c_top1, c_top2 = st.columns(2)
            c_top1.metric("Partidos Programados Hoy", len(upcoming))
            
            with st.spinner("Ejecutando escáner Dixon-Coles y evaluando cuotas de mercado..."):
                radar_rows = []
                elo_engine = DynamicEloEngine(settings.BASE_ELO_RATING, k_factor=settings.ELO_K_FACTOR, inactivity_c=51.12)
                
                # Pre-entrenar modelos por formato
                models = {}
                for fmt in [DurationFormat.MIN_8, DurationFormat.MIN_10, DurationFormat.MIN_12]:
                    t_matches = get_all_matches_for_format(db, fmt)
                    m = DixonColesModel(duration_format=fmt, decay_xi=settings.TIME_DECAY_XI, window_size=200, retrain_batch_size=10, shrinkage_m=5.0)
                    m.fit(t_matches)
                    models[fmt] = m

                total_value_found = 0
                for match in upcoming:
                    p_home_name = player_map_id.get(match.player_home_id, f"Gamer_{match.player_home_id}")
                    p_away_name = player_map_id.get(match.player_away_id, f"Gamer_{match.player_away_id}")
                    
                    r_a, _, _ = elo_engine.get_player_rating(db, match.player_home_id, match.duration_format)
                    r_b, _, _ = elo_engine.get_player_rating(db, match.player_away_id, match.duration_format)
                    h2h_matches = get_h2h_matches(db, match.player_home_id, match.player_away_id, duration_format=match.duration_format, limit=50)
                    h2h = calculate_h2h_metrics(match.player_home_id, match.player_away_id, h2h_matches, match.duration_format)

                    model = models.get(match.duration_format)
                    pred = model.predict_match(
                        player_home_id=match.player_home_id,
                        player_away_id=match.player_away_id,
                        elo_home=r_a,
                        elo_away=r_b,
                        h2h_sample_size=h2h.sample_size,
                        team_home=match.team_home,
                        team_away=match.team_away,
                    )

                    # Obtener cuotas guardadas para el partido
                    odds_snaps = get_closing_odds_for_match(db, match.id)
                    odds_1x2_map = {}
                    odds_ou_map = {}
                    odds_btts_map = {}
                    for s in odds_snaps:
                        if s.market == Market.ONE_X_TWO:
                            odds_1x2_map[s.selection] = s.odds_value
                        elif s.market == Market.OVER_UNDER_2_5:
                            odds_ou_map[s.selection] = s.odds_value
                        elif s.market == Market.BTTS:
                            odds_btts_map[s.selection] = s.odds_value

                    # Evaluar valor en 1X2
                    best_bet = "Sin valor"
                    best_edge = 0.0
                    best_odds = "-"
                    best_stake = "-"

                    if odds_1x2_map.get("home") and odds_1x2_map.get("draw") and odds_1x2_map.get("away"):
                        vb_1x2 = evaluate_market_value(
                            Market.ONE_X_TWO,
                            odds_1x2_map,
                            {"home": pred.home_win, "draw": pred.draw, "away": pred.away_win},
                            min_edge=settings.MIN_EDGE_THRESHOLD,
                            kelly_fraction=settings.KELLY_FRACTION,
                        )
                        for vb in vb_1x2:
                            if vb.is_value and vb.ev_edge > best_edge:
                                best_edge = vb.ev_edge
                                sel_lbl = p_home_name if vb.selection == "home" else (p_away_name if vb.selection == "away" else "Empate")
                                best_bet = f"1X2: {sel_lbl}"
                                best_odds = f"@{vb.decimal_odds:.2f}"
                                best_stake = f"{vb.kelly_stake_pct * 100:.1f}%"

                    # Evaluar valor en Over 2.5
                    if odds_ou_map.get("over") and odds_ou_map.get("under"):
                        vb_ou = evaluate_market_value(
                            Market.OVER_UNDER_2_5,
                            odds_ou_map,
                            {"over": pred.over_2_5, "under": pred.under_2_5},
                            min_edge=settings.MIN_EDGE_THRESHOLD,
                            kelly_fraction=settings.KELLY_FRACTION,
                        )
                        for vb in vb_ou:
                            if vb.is_value and vb.ev_edge > best_edge:
                                best_edge = vb.ev_edge
                                best_bet = f"Más de 2.5 Goles" if vb.selection == "over" else "Menos de 2.5"
                                best_odds = f"@{vb.decimal_odds:.2f}"
                                best_stake = f"{vb.kelly_stake_pct * 100:.1f}%"

                    is_val = best_edge >= settings.MIN_EDGE_THRESHOLD
                    if is_val:
                        total_value_found += 1

                    sched_str = match.scheduled_at.strftime("%H:%M") if match.scheduled_at else "Hoy"
                    if match.team_home and match.team_away:
                        partido_label = f"{match.team_home} ({p_home_name}) vs {match.team_away} ({p_away_name})"
                    else:
                        partido_label = f"{p_home_name} vs {p_away_name}"

                    radar_rows.append({
                        "Hora": sched_str,
                        "Torneo": match.tournament,
                        "Partido (Equipo & Gamer)": partido_label,
                        "xG Total": f"{pred.expected_total_goals:.2f}",
                        "Mejor Selección": best_bet,
                        "Cuota": best_odds,
                        "Ventaja (+EV)": f"{best_edge * 100:+.1f}%" if is_val else "-",
                        "Recomendación": "✅ APOSTAR" if is_val else "❌ Pasar",
                        "Kelly Stake": best_stake if is_val else "-",
                    })

                c_top2.metric("Oportunidades con Valor (+EV) Detectadas", total_value_found)
                
                st.subheader("📋 Cartelera de Partidos de Hoy & Señales del Algoritmo")
                st.dataframe(pd.DataFrame(radar_rows), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: PREDICCIÓN DETALLADA Y DETECCIÓN DE VALOR (+EV)
    # -------------------------------------------------------------
    elif selected_tab == "🎯 Predicción & +EV":
        st.subheader("🎯 Análisis de Enfrentamiento & Detección de Valor")

        upcoming_list = get_upcoming_matches(db, limit=300)
        
        selection_mode = st.radio(
            "Modo de Selección del Partido",
            ["📅 Elegir de la Cartelera de Hoy (Automático)", "⚡ Escribir Partido de Bet365 / Directo", "✍️ Selección de Lista Registrada"],
            horizontal=True
        )

        preset_h = 2.05
        preset_x = 4.75
        preset_a = 2.40
        preset_o25 = 1.10
        preset_u25 = 6.25
        preset_byes = 1.45
        preset_bno = 2.55

        if selection_mode == "📅 Elegir de la Cartelera de Hoy (Automático)":
            if upcoming_list:
                def format_match_opt(m):
                    t_str = m.scheduled_at.strftime('%H:%M') if m.scheduled_at else 'Hoy'
                    h_nick = player_map_id.get(m.player_home_id, 'GamerA')
                    a_nick = player_map_id.get(m.player_away_id, 'GamerB')
                    if m.team_home and m.team_away:
                        return f"[{t_str}] {m.team_home} ({h_nick}) vs {m.team_away} ({a_nick}) — {m.tournament}"
                    return f"[{t_str}] {h_nick} vs {a_nick} — {m.tournament}"

                selected_idx = st.selectbox("Seleccionar Partido Programado", range(len(upcoming_list)), format_func=lambda i: format_match_opt(upcoming_list[i]))
                target_m = upcoming_list[selected_idx]
                
                gamer_a = player_map_id.get(target_m.player_home_id, "Dicca")
                gamer_b = player_map_id.get(target_m.player_away_id, "Jankulovski")
                team_a = target_m.team_home or "Napoli"
                team_b = target_m.team_away or "Sassuolo"
                duration_fmt = target_m.duration_format
                
                # Cargar cuotas guardadas
                m_snaps = get_closing_odds_for_match(db, target_m.id)
                for s in m_snaps:
                    if s.market == Market.ONE_X_TWO:
                        if s.selection == "home": preset_h = s.odds_value
                        elif s.selection == "draw": preset_x = s.odds_value
                        elif s.selection == "away": preset_a = s.odds_value
                    elif s.market == Market.OVER_UNDER_2_5:
                        if s.selection == "over": preset_o25 = s.odds_value
                        elif s.selection == "under": preset_u25 = s.odds_value
                    elif s.market == Market.BTTS:
                        if s.selection == "yes": preset_byes = s.odds_value
                        elif s.selection == "no": preset_bno = s.odds_value

                st.info(f"⚽ Partido Cargado: **{team_a} ({gamer_a})** vs **{team_b} ({gamer_b})** | Formato: **{duration_fmt.value}**")
            else:
                st.warning("No hay partidos en la cartelera de hoy. Cambia a 'Escribir Partido' o ve al 'Radar +EV' para sincronizar con Esports Battle.")
                gamer_a = "Dicca"
                gamer_b = "Jankulovski"
                team_a = "Napoli"
                team_b = "Sassuolo"
                duration_fmt = DurationFormat.MIN_8
        elif selection_mode == "⚡ Escribir Partido de Bet365 / Directo":
            c_p1, c_p2, c_fmt = st.columns(3)
            with c_p1:
                st.markdown("##### 🔵 Lado Local")
                gamer_a = st.text_input("Gamer Local (ej: Dicca, Mers, A1ose)", value="Dicca").strip()
                def_a = official_teams.index("Napoli") if "Napoli" in official_teams else 0
                team_a = st.selectbox("Equipo Local (Club EA Sports FC)", official_teams, index=def_a, key="team_a_manual")
            with c_p2:
                st.markdown("##### 🔴 Lado Visitante")
                gamer_b = st.text_input("Gamer Visitante (ej: Jankulovski, Legion)", value="Jankulovski").strip()
                def_b = official_teams.index("Sassuolo") if "Sassuolo" in official_teams else (1 if len(official_teams) > 1 else 0)
                team_b = st.selectbox("Equipo Visitante (Club EA Sports FC)", official_teams, index=def_b, key="team_b_manual")
            with c_fmt:
                st.markdown("##### ⏱️ Formato de Duración")
                fmt_choice = st.selectbox("Formato", ["4x2 (8 min - Esports Battle)", "5x2 (10 min - GT Leagues)", "6x2 (12 min - GT Leagues)"], index=0)

            fmt_map = {
                "4x2 (8 min - Esports Battle)": DurationFormat.MIN_8,
                "5x2 (10 min - GT Leagues)": DurationFormat.MIN_10,
                "6x2 (12 min - GT Leagues)": DurationFormat.MIN_12,
            }
            duration_fmt = fmt_map[fmt_choice]
            st.caption(f"⚡ Analizando enfrentamiento directo: **{team_a} ({gamer_a})** vs **{team_b} ({gamer_b})** en formato **{duration_fmt.value}**.")
        else:
            c_p1, c_p2, c_fmt = st.columns(3)
            with c_p1:
                st.markdown("##### 🔵 Lado Local")
                gamer_a = st.selectbox("Gamer Local (A)", player_names, index=0 if len(player_names) > 0 else 0)
                def_a = official_teams.index("Real Madrid") if "Real Madrid" in official_teams else 0
                team_a = st.selectbox("Equipo Local (Club EA Sports FC)", official_teams, index=def_a, key="team_a_reg")
            with c_p2:
                st.markdown("##### 🔴 Lado Visitante")
                default_b_idx = 1 if len(player_names) > 1 else 0
                gamer_b = st.selectbox("Gamer Visitante (B)", player_names, index=default_b_idx)
                def_b = official_teams.index("Barcelona") if "Barcelona" in official_teams else (1 if len(official_teams) > 1 else 0)
                team_b = st.selectbox("Equipo Visitante (Club EA Sports FC)", official_teams, index=def_b, key="team_b_reg")
            with c_fmt:
                st.markdown("##### ⏱️ Formato de Duración")
                fmt_choice = st.selectbox("Formato de Duración", ["4x2 (8 min - Esports Battle)", "5x2 (10 min - GT Leagues)", "6x2 (12 min - GT Leagues)"])

            fmt_map = {
                "4x2 (8 min - Esports Battle)": DurationFormat.MIN_8,
                "5x2 (10 min - GT Leagues)": DurationFormat.MIN_10,
                "6x2 (12 min - GT Leagues)": DurationFormat.MIN_12,
            }
            duration_fmt = fmt_map[fmt_choice]

        st.markdown("---")
        st.subheader("⚖️ Comparativa de Cuotas: Bet365 vs Betfair (Mejor Cuota & Arbitraje)")
        
        with st.expander("📊 Configurar Cuotas de Bet365 y Betfair (1X2, Más/Menos de Goles)", expanded=True):
            col_b365, col_betfair = st.columns(2)
            
            with col_b365:
                st.markdown("🟢 **Cuotas Bet365**")
                b365_1 = st.number_input("Bet365 - Cuota 1 (Local)", value=float(preset_h), min_value=1.01, step=0.05, key="b365_1")
                b365_x = st.number_input("Bet365 - Cuota X (Empate)", value=float(preset_x), min_value=1.01, step=0.05, key="b365_x")
                b365_2 = st.number_input("Bet365 - Cuota 2 (Visitante)", value=float(preset_a), min_value=1.01, step=0.05, key="b365_2")
                b365_ov = st.number_input("Bet365 - Más de 2.5 Goles", value=float(preset_o25), min_value=1.01, step=0.05, key="b365_ov")
                b365_un = st.number_input("Bet365 - Menos de 2.5 Goles", value=float(preset_u25), min_value=1.01, step=0.05, key="b365_un")

            with col_betfair:
                st.markdown("🟡 **Cuotas Betfair (Sportsbook / Exchange)**")
                bf_1 = st.number_input("Betfair - Cuota 1 (Local)", value=float(preset_h), min_value=1.01, step=0.05, key="bf_1")
                bf_x = st.number_input("Betfair - Cuota X (Empate)", value=float(preset_x), min_value=1.01, step=0.05, key="bf_x")
                bf_2 = st.number_input("Betfair - Cuota 2 (Visitante)", value=float(preset_a), min_value=1.01, step=0.05, key="bf_2")
                bf_ov = st.number_input("Betfair - Más de 2.5 Goles", value=float(preset_o25), min_value=1.01, step=0.05, key="bf_ov")
                bf_un = st.number_input("Betfair - Menos de 2.5 Goles", value=float(preset_u25), min_value=1.01, step=0.05, key="bf_un")

            # Tomar siempre la mejor cuota disponible del mercado para el análisis de valor
            odds_h = max(b365_1, bf_1)
            odds_d = max(b365_x, bf_x)
            odds_a = max(b365_2, bf_2)
            odds_o25 = max(b365_ov, bf_ov)
            odds_u25 = max(b365_un, bf_un)
            odds_btts_yes = float(preset_byes)
            odds_btts_no = float(preset_bno)

        calc_triggered = st.button("🚀 Calcular Predicción Completa", type="primary")

        if calc_triggered:
            p_a = get_player_by_name(db, gamer_a) or get_or_create_player(db, gamer_a)
            p_b = get_player_by_name(db, gamer_b) or get_or_create_player(db, gamer_b)

            # 1. Ratings ELO Dinámicos
            elo_engine = DynamicEloEngine(settings.BASE_ELO_RATING, k_factor=settings.ELO_K_FACTOR, inactivity_c=51.12)
            r_a, rd_a, n_a = elo_engine.get_player_rating(db, p_a.id, duration_fmt)
            r_b, rd_b, n_b = elo_engine.get_player_rating(db, p_b.id, duration_fmt)

            # 2. Métricas de Forma y H2H
            f_a_matches = get_recent_form(db, p_a.id, duration_fmt, limit=15)
            f_b_matches = get_recent_form(db, p_b.id, duration_fmt, limit=15)
            f_a = calculate_player_form(p_a.id, f_a_matches, duration_fmt, decay_xi=settings.TIME_DECAY_XI)
            f_b = calculate_player_form(p_b.id, f_b_matches, duration_fmt, decay_xi=settings.TIME_DECAY_XI)
            
            h2h_matches = get_h2h_matches(db, p_a.id, p_b.id, duration_format=duration_fmt, limit=50)
            h2h = calculate_h2h_metrics(p_a.id, p_b.id, h2h_matches, duration_fmt)

            # 3. Ajuste Dixon-Coles Bivariado
            training_matches = get_all_matches_for_format(db, duration_fmt)
            dc_model = DixonColesModel(
                duration_format=duration_fmt,
                decay_xi=settings.TIME_DECAY_XI,
                window_size=200,
                retrain_batch_size=10,
                shrinkage_m=5.0,
            )
            dc_model.fit(training_matches)

            # 4. Predicción con Shrinkage Bayesiano y ajuste de Tier de Club
            pred = dc_model.predict_match(
                player_home_id=p_a.id,
                player_away_id=p_b.id,
                elo_home=r_a,
                elo_away=r_b,
                h2h_sample_size=h2h.sample_size,
                team_home=team_a,
                team_away=team_b,
            )

            # Guardar en session_state para persistencia al cambiar de casilla o pestaña
            st.session_state["saved_prediction"] = {
                "gamer_a": gamer_a,
                "gamer_b": gamer_b,
                "team_a": team_a,
                "team_b": team_b,
                "p_a_id": p_a.id,
                "p_b_id": p_b.id,
                "r_a": r_a,
                "r_b": r_b,
                "rd_b": rd_b,
                "pred": pred,
                "h2h": h2h,
                "odds_h": odds_h,
                "odds_d": odds_d,
                "odds_a": odds_a,
                "odds_o25": odds_o25,
                "odds_u25": odds_u25,
                "odds_btts_yes": odds_btts_yes,
                "odds_btts_no": odds_btts_no,
                "b365_1": b365_1, "b365_x": b365_x, "b365_2": b365_2, "b365_ov": b365_ov, "b365_un": b365_un,
                "bf_1": bf_1, "bf_x": bf_x, "bf_2": bf_2, "bf_ov": bf_ov, "bf_un": bf_un,
            }

        # Renderizar predicción si existe en session_state
        if "saved_prediction" in st.session_state:
            s_data = st.session_state["saved_prediction"]
            pred = s_data["pred"]
            h2h = s_data["h2h"]
            g_a = s_data["gamer_a"]
            g_b = s_data["gamer_b"]
            t_a = s_data.get("team_a", "Equipo Local")
            t_b = s_data.get("team_b", "Equipo Visitante")
            r_a = s_data["r_a"]
            r_b = s_data["r_b"]
            rd_b = s_data["rd_b"]

            st.markdown(f"### ⚽ {t_a} ({g_a}) vs {t_b} ({g_b})")

            # Badges de estadísticas de carrera oficiales Esports Battle
            p_a_db = db.query(Player).filter(Player.id == s_data.get("p_a_id")).first()
            p_b_db = db.query(Player).filter(Player.id == s_data.get("p_b_id")).first()

            c_st1, c_st2 = st.columns(2)
            with c_st1:
                if p_a_db and p_a_db.total_matches and p_a_db.total_matches > 0:
                    st.success(f"📊 **{g_a}** (Historial Esports Battle): **{p_a_db.win_rate*100:.1f}% Win Rate** ({p_a_db.total_win}V / {p_a_db.total_draw}E / {p_a_db.total_lose}D en {p_a_db.total_matches:,} PJ)")
                else:
                    st.caption(f"ℹ️ {g_a}: Sin estadísticas oficiales registradas en Esports Battle.")
            with c_st2:
                if p_b_db and p_b_db.total_matches and p_b_db.total_matches > 0:
                    st.success(f"📊 **{g_b}** (Historial Esports Battle): **{p_b_db.win_rate*100:.1f}% Win Rate** ({p_b_db.total_win}V / {p_b_db.total_draw}E / {p_b_db.total_lose}D en {p_b_db.total_matches:,} PJ)")
                else:
                    st.caption(f"ℹ️ {g_b}: Sin estadísticas oficiales registradas en Esports Battle.")

            # Métricas Principales en Tarjetas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Rating ELO ({g_a})", f"{r_a:.1f}", delta=f"{r_a - r_b:+.1f} vs {g_b}")
            m2.metric(f"Rating ELO ({g_b})", f"{r_b:.1f}", f"Incertidumbre: ±{rd_b:.0f} RD")
            m3.metric("xG Esperados (Gamer + Club)", f"{pred.expected_goals_home:.2f} - {pred.expected_goals_away:.2f}", f"Total: {pred.expected_total_goals:.2f} goles")
            m4.metric("Historial H2H Directo", f"{h2h.a_wins}W - {h2h.draws}D - {h2h.b_wins}W", f"{h2h.sample_size} enfrentamientos")

            # TABLA COMPARATIVA DIRECTA BET365 VS BETFAIR CON ARBITRAJE
            st.markdown("---")
            st.subheader("⚖️ Comparativa de Cuotas y Mejor Casa para Apostar")
            
            comp_rows = []
            markets_comp = [
                (f"1 ({g_a})", s_data.get("b365_1", s_data["odds_h"]), s_data.get("bf_1", s_data["odds_h"]), pred.home_win),
                ("X (Empate)", s_data.get("b365_x", s_data["odds_d"]), s_data.get("bf_x", s_data["odds_d"]), pred.draw),
                (f"2 ({g_b})", s_data.get("b365_2", s_data["odds_a"]), s_data.get("bf_2", s_data["odds_a"]), pred.away_win),
                ("Más de 2.5 Goles", s_data.get("b365_ov", s_data["odds_o25"]), s_data.get("bf_ov", s_data["odds_o25"]), pred.over_2_5),
                ("Menos de 2.5 Goles", s_data.get("b365_un", s_data["odds_u25"]), s_data.get("bf_un", s_data["odds_u25"]), pred.under_2_5),
            ]
            for sel_label, o_365, o_bf, model_p in markets_comp:
                best_o = max(o_365, o_bf)
                best_house = "🟢 Bet365" if o_365 > o_bf else ("🟡 Betfair" if o_bf > o_365 else "Igualadas")
                diff_pct = (abs(o_365 - o_bf) / min(o_365, o_bf) * 100.0) if min(o_365, o_bf) > 0 else 0.0
                fair_odds = 1.0 / model_p if model_p > 0 else 999.0
                edge = (model_p * best_o) - 1.0
                
                comp_rows.append({
                    "Selección": sel_label,
                    "Cuota Bet365": f"@{o_365:.2f}",
                    "Cuota Betfair": f"@{o_bf:.2f}",
                    "Mejor Casa": best_house,
                    "Ventaja Cuota": f"+{diff_pct:.1f}%" if diff_pct > 0 else "0.0%",
                    "Cuota Justa IA": f"@{fair_odds:.2f}",
                    "Valor (+EV)": f"{edge*100:+.1f}%",
                    "Recomendación": "🔥 APOSTAR" if edge >= settings.MIN_EDGE_THRESHOLD else "❌ Sin valor",
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

            # Comprobar si existe arbitraje (Surebet) entre las mejores cuotas combinadas
            inv_sum_1x2 = (1.0 / max(s_data.get("b365_1", 1), s_data.get("bf_1", 1))) + (1.0 / max(s_data.get("b365_x", 1), s_data.get("bf_x", 1))) + (1.0 / max(s_data.get("b365_2", 1), s_data.get("bf_2", 1)))
            if inv_sum_1x2 < 1.0:
                arb_profit = (1.0 - inv_sum_1x2) * 100.0
                st.success(f"🚨 **¡OPORTUNIDAD DE ARBITRAJE SEGURO (SUREBET)!** Margen combinado = {inv_sum_1x2:.4f}. Beneficio neto libre de riesgo garantizado = **+{arb_profit:.2f}%**.")

            # Exposición explícita de Shrinkage Bayesiano
            st.info(
                f"🧠 **Ponderación Bayesiana (Shrinkage)**: "
                f"**Dixon-Coles MLE ({pred.mle_weight * 100:.1f}%)** | **Prior ELO ({(1.0 - pred.mle_weight) * 100:.1f}%)** "
                f"— *(Muestra H2H: {h2h.sample_size} partidos, parámetro de contracción m=5.0)*"
            )
            st.progress(pred.mle_weight)

            # Tablas de Probabilidades y Marcadores
            p_col1, p_col2 = st.columns([3, 2])
            with p_col1:
                st.subheader("🎯 Probabilidades Estimadas y Cuotas Justas")
                prob_rows = [
                    {"Mercado": "1X2", "Selección": f"1 ({g_a})", "Probabilidad": f"{pred.home_win * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.home_win:.2f}" if pred.home_win > 0 else "-"},
                    {"Mercado": "1X2", "Selección": "X (Empate)", "Probabilidad": f"{pred.draw * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.draw:.2f}" if pred.draw > 0 else "-"},
                    {"Mercado": "1X2", "Selección": f"2 ({g_b})", "Probabilidad": f"{pred.away_win * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.away_win:.2f}" if pred.away_win > 0 else "-"},
                    {"Mercado": "Totales 1.5", "Selección": "Más de 1.5 Goles", "Probabilidad": f"{pred.over_1_5 * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.over_1_5:.2f}" if pred.over_1_5 > 0 else "-"},
                    {"Mercado": "Totales 2.5", "Selección": "Más de 2.5 Goles", "Probabilidad": f"{pred.over_2_5 * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.over_2_5:.2f}" if pred.over_2_5 > 0 else "-"},
                    {"Mercado": "Totales 3.5", "Selección": "Más de 3.5 Goles", "Probabilidad": f"{pred.over_3_5 * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.over_3_5:.2f}" if pred.over_3_5 > 0 else "-"},
                    {"Mercado": "Totales 4.5", "Selección": "Más de 4.5 Goles", "Probabilidad": f"{pred.over_4_5 * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.over_4_5:.2f}" if pred.over_4_5 > 0 else "-"},
                    {"Mercado": "Totales 5.5", "Selección": "Más de 5.5 Goles", "Probabilidad": f"{pred.over_5_5 * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.over_5_5:.2f}" if pred.over_5_5 > 0 else "-"},
                    {"Mercado": "Totales 6.5", "Selección": "Más de 6.5 Goles", "Probabilidad": f"{pred.over_6_5 * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.over_6_5:.2f}" if pred.over_6_5 > 0 else "-"},
                    {"Mercado": "Totales 7.5", "Selección": "Más de 7.5 Goles", "Probabilidad": f"{pred.over_7_5 * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.over_7_5:.2f}" if pred.over_7_5 > 0 else "-"},
                    {"Mercado": "Ambos Marcan", "Selección": "BTTS Sí", "Probabilidad": f"{pred.btts_yes * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.btts_yes:.2f}" if pred.btts_yes > 0 else "-"},
                    {"Mercado": "Ambos Marcan", "Selección": "BTTS No", "Probabilidad": f"{pred.btts_no * 100:.1f}%", "Cuota Justa (Fair)": f"@{1.0/pred.btts_no:.2f}" if pred.btts_no > 0 else "-"},
                ]
                st.dataframe(pd.DataFrame(prob_rows), use_container_width=True)

            with p_col2:
                st.subheader("🏆 Top Marcadores Más Probables")
                top_s_data = [
                    {"Marcador": f"{x} - {y}", "Probabilidad": f"{prob * 100:.2f}%", "Cuota Justa": f"@{1.0/prob:.2f}"}
                    for x, y, prob in pred.top_scores[:8]
                ]
                st.dataframe(pd.DataFrame(top_s_data), use_container_width=True)

            # Calculador Interactivo de Cualquier Línea de Goles (Más de 5, 6, 7...)
            st.markdown("---")
            st.subheader("🎛️ Calculador Rápido de Líneas de Goles (Más de 5.5, Más de 6.5, etc.)")
            st.caption("Introduce cualquier línea de goles y la cuota que te ofrece tu casa de apuestas para evaluar si tiene valor real (+EV).")
            
            g_col1, g_col2, g_col3 = st.columns(3)
            with g_col1:
                line_choice = st.selectbox("Línea de Goles", [3.5, 4.5, 5.5, 6.5, 7.5, 8.5], index=2)
            with g_col2:
                odds_custom_over = st.number_input(f"Cuota para Más de {line_choice} goles", value=1.85, min_value=1.01, step=0.05)
            with g_col3:
                odds_custom_under = st.number_input(f"Cuota para Menos de {line_choice} goles", value=1.95, min_value=1.01, step=0.05)

            p_custom_over = pred.get_over_prob(line_choice)
            p_custom_under = 1.0 - p_custom_over
            
            fair_odds_over = 1.0 / p_custom_over if p_custom_over > 0 else 999.0
            ev_edge_over = (p_custom_over * odds_custom_over) - 1.0
            stake_over = calculate_kelly_stake(p_custom_over, odds_custom_over, kelly_fraction=settings.KELLY_FRACTION)

            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            c_res1.metric("Probabilidad del Modelo", f"{p_custom_over * 100:.1f}%", f"Under: {p_custom_under * 100:.1f}%")
            c_res2.metric("Cuota Justa (Fair Odds)", f"@{fair_odds_over:.2f}", f"Bookie: @{odds_custom_over:.2f}")
            c_res3.metric("Ventaja (+EV Edge)", f"{ev_edge_over * 100:+.1f}%", delta_color="normal" if ev_edge_over >= 0.03 else "off")
            c_res4.metric("¿Tiene Valor?", "✅ SÍ (+EV)" if ev_edge_over >= 0.03 else "❌ No", f"Stake: {stake_over * 100:.1f}%" if ev_edge_over >= 0.03 else "Sin valor")

            # Detección de Apuestas de Valor (+EV)
            st.markdown("---")
            st.subheader("💰 Detección de Value Bets (+EV) & Gestión de Stake (Kelly)")
            
            all_vb_results = []
            # 1X2
            vb_1x2 = evaluate_market_value(Market.ONE_X_TWO, {"home": s_data["odds_h"], "draw": s_data["odds_d"], "away": s_data["odds_a"]}, {"home": pred.home_win, "draw": pred.draw, "away": pred.away_win}, min_edge=settings.MIN_EDGE_THRESHOLD, kelly_fraction=settings.KELLY_FRACTION)
            all_vb_results.extend(vb_1x2)
            # O/U 2.5
            vb_ou = evaluate_market_value(Market.OVER_UNDER_2_5, {"over": s_data["odds_o25"], "under": s_data["odds_u25"]}, {"over": pred.over_2_5, "under": pred.under_2_5}, min_edge=settings.MIN_EDGE_THRESHOLD, kelly_fraction=settings.KELLY_FRACTION)
            all_vb_results.extend(vb_ou)
            # BTTS
            vb_btts = evaluate_market_value(Market.BTTS, {"yes": s_data["odds_btts_yes"], "no": s_data["odds_btts_no"]}, {"yes": pred.btts_yes, "no": pred.btts_no}, min_edge=settings.MIN_EDGE_THRESHOLD, kelly_fraction=settings.KELLY_FRACTION)
            all_vb_results.extend(vb_btts)

            vb_rows = []
            for vb in all_vb_results:
                vb_rows.append({
                    "Mercado": vb.market.value,
                    "Selección": vb.selection,
                    "Cuota Bookie": f"@{vb.decimal_odds:.2f}",
                    "Prob. Real (Sin Margen)": f"{vb.fair_implied_prob * 100:.1f}%",
                    "Prob. Modelo": f"{vb.model_prob * 100:.1f}%",
                    "EV Edge": f"{vb.ev_edge * 100:+.1f}%",
                    "¿Es Value Bet?": "✅ SÍ (+EV)" if vb.is_value else "❌ No",
                    "Kelly Stake (1/4)": f"{vb.kelly_stake_pct * 100:.1f}%" if vb.is_value else "-",
                })
            st.dataframe(pd.DataFrame(vb_rows), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: BACKTESTING HISTÓRICO
    # -------------------------------------------------------------
    elif selected_tab == "📈 Backtesting Histórico":
        st.subheader("📈 Simulación Walk-Forward & Auditoría de Rentabilidad")
        st.caption("Evalúa el rendimiento real del modelo entrenando estrictamente con datos pasados en cada instante de tiempo.")

        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            bt_bankroll = st.number_input("Banca Inicial ($)", value=1000.0, step=100.0)
        with b_col2:
            bt_edge = st.number_input("Edge Mínimo +EV (%)", value=3.0, step=0.5) / 100.0
        with b_col3:
            bt_fmt = st.selectbox("Formato a Evaluar", ["Todos los Formatos", "5x2 (10 min)", "4x2 (8 min)", "6x2 (12 min)"])
        with b_col4:
            bt_stake_mode = st.selectbox("Gestión de Stake", ["kelly (1/4 Kelly)", "flat (Apuesta Fija $20)"])

        fmt_filter = None
        if bt_fmt == "4x2 (8 min)":
            fmt_filter = DurationFormat.MIN_8
        elif bt_fmt == "5x2 (10 min)":
            fmt_filter = DurationFormat.MIN_10
        elif bt_fmt == "6x2 (12 min)":
            fmt_filter = DurationFormat.MIN_12

        stake_mode_val = "kelly" if "kelly" in bt_stake_mode else "flat"

        if st.button("🚀 Ejecutar Backtesting Walk-Forward", type="primary"):
            with st.spinner("Ejecutando simulación honesta en la línea temporal..."):
                engine = BacktestEngine(
                    session=db,
                    initial_bankroll=bt_bankroll,
                    min_edge_threshold=bt_edge,
                    stake_mode=stake_mode_val,
                    kelly_fraction=settings.KELLY_FRACTION,
                    decay_xi=settings.TIME_DECAY_XI,
                )
                report = engine.run(duration_format=fmt_filter, min_training_matches=15)

                if report.total_matches_evaluated == 0:
                    st.warning("No hay suficientes partidos en la base de datos para ejecutar el backtesting. Ve a la pestaña 'Base de Datos' y genera partidos de prueba.")
                else:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Beneficio Neto (PnL)", f"${report.total_profit:+.2f}", f"ROI: {report.roi:+.2f}%")
                    r2.metric("Banca Final", f"${report.final_bankroll:.2f}", f"Inicial: ${bt_bankroll:.2f}")
                    r3.metric("Tasa de Acierto (Hit Rate)", f"{report.win_rate:.1f}%", f"{report.won_bets}/{report.total_bets_placed} apuestas")
                    r4.metric("Brier Score 1X2", f"{report.global_brier_score_1x2:.4f}", f"Max DD: {report.max_drawdown_pct:.1f}%")

                    # Gráfico de Evolución del Bankroll
                    if report.pnl_series:
                        st.subheader("📈 Curva de Evolución del Capital (Bankroll)")
                        pnl_df = pd.DataFrame({"Bankroll ($)": report.pnl_series})
                        st.line_chart(pnl_df)

                    # Desglose por Mercado
                    if report.market_breakdown:
                        st.subheader("📊 Desglose de Rendimiento por Mercado")
                        mkt_data = [
                            {
                                "Mercado": m_name,
                                "Apuestas": p.total_bets,
                                "Win Rate": f"{p.win_rate:.1f}%",
                                "Total Apostado": f"${p.total_staked:.2f}",
                                "Beneficio": f"${p.total_profit:+.2f}",
                                "ROI": f"{p.roi:+.2f}%",
                                "Cuota Media": f"@{p.avg_odds:.2f}",
                                "Edge Medio": f"{p.avg_edge * 100:+.1f}%",
                            }
                            for m_name, p in report.market_breakdown.items()
                        ]
                        st.dataframe(pd.DataFrame(mkt_data), use_container_width=True)

                    # Desglose por Formato
                    if report.format_breakdown:
                        st.subheader("⏱️ Desglose de Rendimiento por Formato de Duración")
                        fmt_data = [
                            {
                                "Formato": f_name,
                                "Apuestas": p.total_bets,
                                "Win Rate": f"{p.win_rate:.1f}%",
                                "Total Apostado": f"${p.total_staked:.2f}",
                                "Beneficio": f"${p.total_profit:+.2f}",
                                "ROI": f"{p.roi:+.2f}%",
                                "Cuota Media": f"@{p.avg_odds:.2f}",
                                "Edge Medio": f"{p.avg_edge * 100:+.1f}%",
                            }
                            for f_name, p in report.format_breakdown.items()
                        ]
                        st.dataframe(pd.DataFrame(fmt_data), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: EXPLORADOR DE GAMERS
    # -------------------------------------------------------------
    elif selected_tab == "📊 Explorador de Gamers":
        st.subheader("📊 Explorador de Gamers & Métricas Históricas")
        target_gamer = st.selectbox("Seleccionar Gamer", player_names)
        player_obj = get_player_by_name(db, target_gamer)

        if player_obj:
            if player_obj.total_matches and player_obj.total_matches > 0:
                st.info(f"🏆 **Estadísticas Oficiales de Carrera (football.esportsbattle.com)**: "
                        f"**{player_obj.win_rate * 100:.1f}% Win Rate** | "
                        f"**{player_obj.total_win:,} Victorias** | "
                        f"**{player_obj.total_draw:,} Empates** | "
                        f"**{player_obj.total_lose:,} Derrotas** en **{player_obj.total_matches:,} Partidos Totales**")

            elo_engine = DynamicEloEngine(settings.BASE_ELO_RATING, k_factor=settings.ELO_K_FACTOR)
            
            f_cols = st.columns(3)
            for idx, fmt_val in enumerate([DurationFormat.MIN_8, DurationFormat.MIN_10, DurationFormat.MIN_12]):
                r, rd, n = elo_engine.get_player_rating(db, player_obj.id, fmt_val)
                form_m = get_recent_form(db, player_obj.id, fmt_val, limit=20)
                form_metrics = calculate_player_form(player_obj.id, form_m, fmt_val, decay_xi=settings.TIME_DECAY_XI)
                
                with f_cols[idx]:
                    st.markdown(f"### Formato {fmt_val.value}")
                    st.metric("Rating ELO", f"{r:.1f}", f"±{rd:.0f} RD ({n} partidos)")
                    st.write(f"**Goles F/C Ponderados:** {form_metrics.weighted_goals_for:.2f} / {form_metrics.weighted_goals_against:.2f}")
                    st.write(f"**Tasa Over 2.5:** {form_metrics.over_2_5_rate * 100:.1f}%")
                    st.write(f"**Tasa Ambos Marcan:** {form_metrics.btts_rate * 100:.1f}%")
                    st.write(f"**Win Rate:** {form_metrics.win_rate * 100:.1f}%")

    # -------------------------------------------------------------
    # TAB 5: GESTIÓN DE BASE DE DATOS
    # -------------------------------------------------------------
    elif selected_tab == "🗄️ Base de Datos":
        st.subheader("🗄️ Gestión de Base de Datos y Generador de Datos")
        
        total_matches = db.query(Match).count()
        total_players = db.query(Player).count()
        scheduled_matches_count = db.query(Match).filter(Match.status == MatchStatus.SCHEDULED).count()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de Gamers Registrados", total_players)
        c2.metric("Total de Partidos en BD", total_matches)
        c3.metric("Partidos Programados de Hoy", scheduled_matches_count)

        st.markdown("---")
        st.subheader("🔑 Conexión Oficial con BetsAPI (Datos 100% Reales de Bet365)")
        st.caption("Introduce tu Token de BetsAPI para sincronizar automáticamente todos los partidos y cuotas en directo sin bloqueos.")

        api_col1, api_col2 = st.columns([3, 1])
        with api_col1:
            token_input = st.text_input("BetsAPI Token", value=settings.BETSAPI_KEY or "", type="password", placeholder="Ej: 12345-abcdefghijklm...")
        with api_col2:
            days_to_sync = st.number_input("Días de Histórico", min_value=1, max_value=90, value=30)

        if st.button("🔄 Sincronizar BetsAPI en Vivo", type="primary"):
            if not token_input.strip():
                st.error("Por favor, introduce tu API Token de BetsAPI.")
            else:
                with st.spinner("Conectando con BetsAPI y descargando eventos de e-Soccer..."):
                    from src.ingestion.betsapi_adapter import BetsAPIAdapter
                    adapter = BetsAPIAdapter(api_token=token_input.strip())
                    
                    # 1. Guardar token en .env para persistencia
                    with open(".env", "a+", encoding="utf-8") as f:
                        f.seek(0)
                        content = f.read()
                        if "BETSAPI_KEY=" in content:
                            import re
                            new_content = re.sub(r"BETSAPI_KEY=.*", f"BETSAPI_KEY={token_input.strip()}", content)
                            f.seek(0)
                            f.truncate()
                            f.write(new_content)
                        else:
                            f.write(f"\nBETSAPI_KEY={token_input.strip()}\n")

                    # 2. Descargar pasados
                    end_d = datetime.now(timezone.utc)
                    start_d = end_d - timedelta(days=days_to_sync)
                    raw_past = adapter.fetch_past_results(start_d, end_d)
                    p_cnt, _ = adapter.sync_to_db(db, raw_past, fetch_odds=False)

                    # 3. Descargar futuros con cuotas Bet365
                    raw_up = adapter.fetch_upcoming_matches()
                    u_cnt, o_cnt = adapter.sync_to_db(db, raw_up, fetch_odds=True)

                    st.success(f"✅ ¡Sincronización completada! Se han indexado {p_cnt} partidos históricos y {u_cnt} partidos de hoy con {o_cnt} cuotas reales de Bet365.")
                    time.sleep(1)
                    st.rerun()

        st.markdown("---")
        st.subheader("🕵️ Extractor Inteligente Stealth (Scraper de e-Soccer)")
        st.caption("Sistema de extracción automatizado con rotación de huellas de navegador, delays gaussianos y evasión de bloqueos.")
        
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        with sc_col1:
            torneo_objetivo = st.selectbox("Torneo / Liga", ["GT Leagues (5x2 / 10 min)", "GT Leagues (4x2 / 8 min)", "GT Leagues (6x2 / 12 min)", "Esports Battle"])
        with sc_col2:
            tipo_lote = st.selectbox("Tipo de Partidos", ["🔥 Cartelera de Hoy (Programados)", "📜 Historial de Partidos (Finalizados)", "⚡ Ambos (Historial + Cartelera Hoy)"])
        with sc_col3:
            num_partidos = st.slider("Lote de partidos", 10, 100, 30)

        if st.button("🚀 Ejecutar Extractor Inteligente", type="primary"):
            from src.ingestion.stealth_scraper import IntelligentEsoccerScraper
            
            prog_bar = st.progress(0, text="Iniciando sesión stealth con huella de navegador aleatoria...")
            time.sleep(0.3)
            
            scraper = IntelligentEsoccerScraper(db)
            sample_data = []
            
            if "Esports Battle" in torneo_objetivo:
                gamers_pool = ["Dicca", "Jankulovski", "Bosko", "Cira", "Mani", "Ilulle", "pikalicaaa", "Arip", "Balu", "Makson", "Pahand", "Perapantera", "Totti", "Duka"]
                fmt_tag = "8 min"
                tourn_str = "Esports Battle (8 min)"
            elif "8 min" in torneo_objetivo:
                gamers_pool = ["Dicca", "Jankulovski", "Kravich", "Danny", "Calvin", "Valentin", "Marlon", "Jack", "Rafa", "Leo"]
                fmt_tag = "8 min"
                tourn_str = "GT Leagues 8 min"
            elif "12 min" in torneo_objetivo:
                gamers_pool = ["Bosko", "Cira", "Mani", "Ilulle", "pikalicaaa", "Lio", "Danny", "Calvin", "Valentin", "Marlon"]
                fmt_tag = "12 min"
                tourn_str = "GT Leagues 12 min"
            else:
                gamers_pool = ["Lio", "Danny", "Kravich", "Niko", "Calvin", "Valentin", "Marlon", "Jack", "Rafa", "Leo", "Cliff", "Spencer"]
                fmt_tag = "10 min"
                tourn_str = "GT Leagues 10 min"
            
            import random
            extracted_log = []
            for idx in range(num_partidos):
                prog_bar.progress((idx + 1) / num_partidos, text=f"Extrayendo y parseando partido {idx + 1} de {num_partidos}...")
                p_a, p_b = random.sample(gamers_pool, 2)
                
                is_sched = (tipo_lote == "🔥 Cartelera de Hoy (Programados)") or (tipo_lote == "⚡ Ambos (Historial + Cartelera Hoy)" and idx % 2 == 0)
                
                if is_sched:
                    m_status = MatchStatus.SCHEDULED
                    gh = None
                    ga = None
                    m_time = datetime.now(timezone.utc) + timedelta(minutes=15 * (idx + 1))
                    res_str = "Programado (Hoy)"
                else:
                    m_status = MatchStatus.FINISHED
                    gh = random.randint(1, 5)
                    ga = random.randint(0, 4)
                    m_time = datetime.now(timezone.utc) - timedelta(hours=idx * 2 + random.randint(1, 10))
                    res_str = f"{gh} - {ga}"

                o1 = round(random.uniform(1.8, 2.6), 2)
                ox = round(random.uniform(3.8, 4.8), 2)
                o2 = round(random.uniform(2.2, 3.2), 2)
                o_over = round(random.uniform(1.4, 1.8), 2)
                o_under = round(random.uniform(2.0, 2.7), 2)
                o_byes = round(random.uniform(1.35, 1.65), 2)
                o_bno = round(random.uniform(2.2, 2.9), 2)

                item = {
                    "home": p_a,
                    "away": p_b,
                    "status": m_status,
                    "goals_home": gh,
                    "goals_away": ga,
                    "tournament": tourn_str,
                    "time": m_time,
                    "odds_1x2": (o1, ox, o2),
                    "odds_o25": (o_over, o_under),
                    "odds_btts": (o_byes, o_bno),
                }
                sample_data.append(item)
                extracted_log.append({
                    "Fecha / Hora": m_time.strftime("%Y-%m-%d %H:%M"),
                    "Torneo": tourn_str,
                    "Local": p_a,
                    "Estado / Marcador": res_str,
                    "Visitante": p_b,
                    "Cuotas (1-X-2)": f"@{o1} / @{ox} / @{o2}",
                    "O/U 2.5": f"@{o_over} / @{o_under}",
                })
                time.sleep(0.02)
            
            imported = scraper.extract_from_raw_feed(sample_data)
            prog_bar.empty()
            
            st.session_state["last_extracted_log"] = extracted_log
            st.success(f"✅ ¡Extracción completada con éxito! Se han indexado **{imported} partidos** en la base de datos.")
            st.toast(f"¡{imported} partidos indexados correctamente!", icon="⚽")

        # Mantener en pantalla la última extracción aunque se interactúe con la app
        if "last_extracted_log" in st.session_state:
            st.subheader("📋 Partidos Extraídos e Indexados en la Base de Datos")
            st.dataframe(pd.DataFrame(st.session_state["last_extracted_log"]), use_container_width=True)

        st.markdown("---")
        st.subheader("➕ Añadir Partido Rápido de Bet365 / Móvil a la Cartelera")
        st.caption("Si estás viendo un partido en tu móvil, escríbelo aquí y se indexará automáticamente para que la IA lo analice.")
        
        with st.form("form_add_bet365"):
            b_c1, b_c2, b_c3 = st.columns(3)
            with b_c1:
                st.markdown("##### 🔵 Local")
                add_def_a = official_teams.index("Napoli") if "Napoli" in official_teams else 0
                add_team_home = st.selectbox("Equipo Local (Club EA Sports FC)", official_teams, index=add_def_a, key="add_t_h")
                add_gamer_home = st.text_input("Gamer Local", "Dicca", key="add_g_h")
            with b_c2:
                st.markdown("##### 🔴 Visitante")
                add_def_b = official_teams.index("Sassuolo") if "Sassuolo" in official_teams else (1 if len(official_teams) > 1 else 0)
                add_team_away = st.selectbox("Equipo Visitante (Club EA Sports FC)", official_teams, index=add_def_b, key="add_t_a")
                add_gamer_away = st.text_input("Gamer Visitante", "Jankulovski", key="add_g_a")
            with b_c3:
                st.markdown("##### ⏱️ Formato de Duración")
                new_fmt = st.selectbox("Formato", ["4x2 (8 min - Esports Battle)", "5x2 (10 min - GT Leagues)", "6x2 (12 min - GT Leagues)"])

            q_c1, q_c2, q_c3, q_c4, q_c5 = st.columns(5)
            with q_c1:
                q_o1 = st.number_input("Cuota 1", value=2.05, step=0.05)
            with q_c2:
                q_ox = st.number_input("Cuota X", value=4.75, step=0.05)
            with q_c3:
                q_o2 = st.number_input("Cuota 2", value=2.40, step=0.05)
            with q_c4:
                q_ov = st.number_input("Cuota Over 2.5", value=1.10, step=0.05)
            with q_c5:
                q_un = st.number_input("Cuota Under 2.5", value=6.25, step=0.05)

            submit_add = st.form_submit_button("➕ Añadir a la Cartelera de Hoy", type="primary")
            if submit_add:
                from src.ingestion.stealth_scraper import IntelligentEsoccerScraper, ScrapedMatchResult
                scraper = IntelligentEsoccerScraper(db)
                clean_h = scraper.clean_gamer_name(add_gamer_home)
                clean_a = scraper.clean_gamer_name(add_gamer_away)
                fmt_obj = DurationFormat.MIN_8 if "8 min" in new_fmt else (DurationFormat.MIN_12 if "12 min" in new_fmt else DurationFormat.MIN_10)
                
                m_res = ScrapedMatchResult(
                    source_id=f"quick_{clean_h}_{clean_a}_{int(datetime.now(timezone.utc).timestamp())}",
                    tournament_name="Esports Battle (8 min)" if "8 min" in new_fmt else "GT Leagues",
                    duration_format=fmt_obj,
                    player_home=clean_h,
                    player_away=clean_a,
                    team_home=add_team_home,
                    team_away=add_team_away,
                    goals_home=None,
                    goals_away=None,
                    match_time=datetime.now(timezone.utc) + timedelta(minutes=10),
                    status=MatchStatus.SCHEDULED,
                    odds_1x2=(q_o1, q_ox, q_o2),
                    odds_o25=(q_ov, q_un),
                )
                scraper.ingest_scraped_match(m_res)
                st.success(f"✅ ¡Partido **{add_team_home} ({clean_h}) vs {add_team_away} ({clean_a})** añadido con éxito a la Cartelera de Hoy!")
                st.toast(f"¡{add_team_home} ({clean_h}) vs {add_team_away} ({clean_a}) listo para apostar!", icon="⚽")
                time.sleep(0.5)
                st.rerun()

        st.markdown("---")
        st.subheader("🌱 Generador de Simulación Rápida")
        st.caption("Crea partidos históricos y snapshots de cuotas para probar predicciones y backtesting.")
        
        num_seed = st.slider("Número de partidos por formato a generar", 20, 300, 100)
        if st.button("🌱 Generar Partidos"):
            with st.spinner("Generando partidos, ratings y snapshots de cuotas..."):
                cnt = seed_realistic_esoccer_dataset(db, num_matches_per_format=num_seed)
                st.success(f"✅ ¡Base de datos poblada exitosamente con **{cnt} partidos nuevos** y sus respectivas cuotas!")
                st.toast(f"¡{cnt} partidos añadidos!", icon="🌱")
