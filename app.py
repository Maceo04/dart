#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import uuid
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from streamlit_gsheets import GSheetsConnection

# -----------------------
# Configuration générale
# -----------------------

st.set_page_config(
    page_title="Classement Elo Fléchettes",
    page_icon="🎯",
    layout="wide"
)

BASE_ELO = 1000.0
K_FACTOR = 32.0

# Connexion Google Sheets (nécessite st-gsheets-connection et secrets)
# Secrets attendus: [connections.gsheets] ... (service account + spreadsheet URL)
conn = st.connection("gsheets", type=GSheetsConnection)

# -----------------------
# Fonctions utilitaires
# -----------------------

def key_name(name: str) -> str:
    return name.strip().lower()

def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


@st.cache_resource
def get_conn():
    from streamlit_gsheets import GSheetsConnection
    return st.connection("gsheets", type=GSheetsConnection)

conn = get_conn()

@st.cache_data(ttl=60)
def load_players_df_cached():
    df = conn.read(worksheet="players", ttl=0)
    if df is None or df.empty:
        df = pd.DataFrame(columns=["key","name","elo"])
    df["elo"] = pd.to_numeric(df.get("elo", pd.Series(dtype=float)), errors="coerce").fillna(1000.0)
    df["key"] = df.get("key", "").astype(str)
    df["name"] = df.get("name", "").astype(str)
    return df

@st.cache_data(ttl=60)
def load_matches_df_cached():
    df = conn.read(worksheet="matches", ttl=0)
    if df is None or df.empty:
        df = pd.DataFrame(columns=["id","timestamp","players_csv","ranking_csv","elo_changes_json"])
    return df

def invalidate_caches():
    load_players_df_cached.clear()
    load_matches_df_cached.clear()

def save_players_df(df):
    df = df.fillna("").astype(str)
    conn.update(worksheet="players", data=df)
    invalidate_caches()

def save_matches_df(df):
    df = df.fillna("").astype(str)
    conn.update(worksheet="matches", data=df)
    invalidate_caches()

import time
from gspread.exceptions import APIError

def with_backoff(fn, *args, **kwargs):
    delay = 1.0
    for _ in range(5):
        try:
            return fn(*args, **kwargs)
        except APIError:
            time.sleep(delay)
            delay *= 2
    # dernière tentative
    return fn(*args, **kwargs)


# -----------------------
# Accès Google Sheets
# -----------------------

def load_players_df():
    df = conn.read(worksheet="players", ttl=0)
    if df is None or df.empty:
        df = pd.DataFrame(columns=["key", "name", "elo"])
    # normaliser types
    if "elo" in df.columns:
        df["elo"] = pd.to_numeric(df["elo"], errors="coerce").fillna(BASE_ELO)
    if "key" in df.columns:
        df["key"] = df["key"].astype(str)
    if "name" in df.columns:
        df["name"] = df["name"].astype(str)
    return df

def save_players_df(df):
    # Remplace l'onglet "players" par le DataFrame fourni
    conn.update(worksheet="players", data=df)

def load_matches_df():
    df = conn.read(worksheet="matches", ttl=0)
    if df is None or df.empty:
        df = pd.DataFrame(columns=["id", "timestamp", "players_csv", "ranking_csv", "elo_changes_json"])
    return df

def save_matches_df(df):
    conn.update(worksheet="matches", data=df)

# -----------------------
# Logique Elo
# -----------------------

def ensure_players_df(df_players, names):
    """Crée les joueurs manquants (BASE_ELO) et met à jour les noms (affichage)."""
    if df_players is None or df_players.empty:
        df_players = pd.DataFrame(columns=["key", "name", "elo"])
    existing = set(df_players["key"]) if not df_players.empty else set()
    for name in names:
        k = key_name(name)
        if k in existing:
            df_players.loc[df_players["key"] == k, "name"] = name.strip()
        else:
            df_players.loc[len(df_players)] = {"key": k, "name": name.strip(), "elo": BASE_ELO}
    save_players_df(df_players)
    return load_players_df()

def update_ratings_multiplayer_from_df(df_players, sel_keys, ranking_keys, K=K_FACTOR):
    """Mise à jour Elo pairwise moyenne pour 2 à 4 joueurs."""
    pos = {k: i for i, k in enumerate(ranking_keys)}
    olds = {}
    for k in sel_keys:
        row = df_players.loc[df_players["key"] == k]
        olds[k] = float(row["elo"].iloc[0]) if not row.empty else BASE_ELO
    n = len(sel_keys)
    news = {}
    for i in sel_keys:
        ri = olds[i]
        sum_diff = 0.0
        for j in sel_keys:
            if i == j:
                continue
            rj = olds[j]
            s_ij = 1.0 if pos[i] < pos[j] else 0.0
            e_ij = expected(ri, rj)
            sum_diff += (s_ij - e_ij)
        delta = K * (sum_diff / (n - 1))
        news[i] = ri + delta
    return olds, news

def record_match_sheet(sel_keys, ranking_keys, olds, news):
    df = load_matches_df()
    changes = {k: {"old": float(olds[k]), "new": float(news[k]), "delta": float(news[k] - olds[k])} for k in sel_keys}
    row = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "players_csv": ",".join(sel_keys),
        "ranking_csv": ",".join(ranking_keys),
        "elo_changes_json": json.dumps(changes, ensure_ascii=False)
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_matches_df(df)
    return row

def apply_elo_updates(df_players, news):
    idx = {k: i for i, k in enumerate(df_players["key"])} if not df_players.empty else {}
    for k, new_elo in news.items():
        i = idx.get(k, None)
        if i is not None:
            df_players.at[i, "elo"] = float(new_elo)
    save_players_df(df_players)
    return load_players_df()

# -----------------------
# Interface
# -----------------------

st.title("🎯 Classement Elo Fléchettes")

menu = st.sidebar.radio(
    "Navigation",
    ["Classement", "Nouvelle Partie", "Ajouter un Joueur", "Historique", "Statistiques", "Graphiques", "Exporter"]
)

if menu == "Classement":
    st.header("Classement des Joueurs")
    df_players = load_players_df()
    if df_players.empty:
        st.info("Aucun joueur enregistré. Ajoutez des joueurs pour commencer !")
    else:
        df_show = df_players.sort_values("elo", ascending=False).reset_index(drop=True)
        for idx_row, row in df_show.iterrows():
            rank = idx_row + 1
            col1, col2, col3 = st.columns([1, 4, 2])
            with col1:
                if rank == 1:
                    st.markdown("🥇")
                elif rank == 2:
                    st.markdown("🥈")
                elif rank == 3:
                    st.markdown("🥉")
                else:
                    st.markdown(f"**{rank}**")
            with col2:
                st.markdown(f"**{row['name']}**")
            with col3:
                st.markdown(f"**{float(row['elo']):.1f}** points")
            st.divider()

elif menu == "Nouvelle Partie":
    st.header("Enregistrer une Nouvelle Partie")
    df_players = load_players_df()
    if df_players.empty or len(df_players) < 2:
        st.warning("Vous devez avoir au moins 2 joueurs pour enregistrer une partie.")
    else:
        player_names = df_players["name"].tolist()

        st.subheader("1. Sélectionnez les joueurs")
        num_players = st.radio("Nombre de joueurs", [2, 3, 4], horizontal=True)

        selected_players = []
        cols = st.columns(num_players)
        for i in range(num_players):
            with cols[i]:
                player = st.selectbox(f"Joueur {i+1}", player_names, key=f"player_{i}")
                selected_players.append(player)

        if len(selected_players) != len(set(selected_players)):
            st.error("Chaque joueur ne peut être sélectionné qu'une seule fois !")
        else:
            st.subheader("2. Définissez le classement")
            st.write(f"Indiquez la position de chaque joueur (1 = premier, {num_players} = dernier)")

            rankings = {}
            for player in selected_players:
                rankings[player] = st.selectbox(
                    f"Position de {player}",
                    list(range(1, num_players + 1)),
                    key=f"rank_{player}"
                )

            ranking_values = list(rankings.values())
            if len(ranking_values) != len(set(ranking_values)):
                st.error("Chaque position doit être attribuée à un seul joueur !")
            else:
                if st.button("Enregistrer la partie", type="primary"):
                    # 1) S'assurer que tous existent / mettre à jour les noms
                    df_players = ensure_players_df(df_players, selected_players)

                    # 2) Dériver les clés et l'ordre
                    sel_keys = [key_name(n) for n in selected_players]
                    sorted_players = sorted(selected_players, key=lambda p: rankings[p])
                    ranking_keys = [key_name(p) for p in sorted_players]

                    # 3) Calcul Elo
                    olds, news = update_ratings_multiplayer_from_df(df_players, sel_keys, ranking_keys, K_FACTOR)

                    # 4) Enregistrer match
                    record_match_sheet(sel_keys, ranking_keys, olds, news)

                    # 5) Appliquer Elo
                    df_players = apply_elo_updates(df_players, news)

                    # 6) Feedback
                    st.success("✅ Partie enregistrée avec succès !")
                    st.subheader("Changements de classement")
                    for player in sorted_players:
                        k = key_name(player)
                        delta = news[k] - olds[k]
                        sign = "+" if delta >= 0 else ""
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{player}**")
                        with col2:
                            color = "green" if delta >= 0 else "red"
                            st.markdown(f":{color}[{olds[k]:.1f} → {news[k]:.1f} ({sign}{delta:.1f})]")
                    st.balloons()

elif menu == "Ajouter un Joueur":
    st.header("Ajouter un Nouveau Joueur")
    df_players = load_players_df()
    new_player_name = st.text_input("Nom du joueur")
    if st.button("Ajouter", type="primary"):
        if not new_player_name.strip():
            st.error("Le nom du joueur ne peut pas être vide !")
        else:
            k = key_name(new_player_name)
            if not df_players.empty and (df_players["key"] == k).any():
                st.warning(f"Le joueur '{new_player_name}' existe déjà dans la base !")
            else:
                # append puis save
                new_row = {"key": k, "name": new_player_name.strip(), "elo": BASE_ELO}
                df_players = pd.concat([df_players, pd.DataFrame([new_row])], ignore_index=True)
                save_players_df(df_players)
                st.success(f"✅ Joueur '{new_player_name}' ajouté avec un Elo de {BASE_ELO} !")
                st.rerun()

elif menu == "Historique":
    st.header("Historique des Parties")
    df_matches = load_matches_df()
    df_players = load_players_df()

    if df_matches.empty:
        st.info("Aucune partie enregistrée pour le moment.")
    else:
        df_sorted = df_matches.sort_values("timestamp", ascending=False).reset_index(drop=True)
        st.write(f"**{len(df_sorted)}** parties jouées au total")

        if st.button("🗑️ Annuler la dernière partie", type="secondary"):
            last = df_sorted.iloc[0].to_dict()
            changes = json.loads(last["elo_changes_json"])
            # Restaurer anciens elos
            for k, v in changes.items():
                df_players.loc[df_players["key"] == k, "elo"] = float(v["old"])
            save_players_df(df_players)
            # Supprimer la ligne de match
            df_new = df_matches[df_matches["id"] != last["id"]].reset_index(drop=True)
            save_matches_df(df_new)
            st.success("✅ Dernière partie annulée avec succès !")
            st.rerun()

        st.divider()

        for idx, row in df_sorted.iterrows():
            try:
                ts = datetime.fromisoformat(row["timestamp"])
            except Exception:
                ts = datetime.utcnow()
            ranking_keys = row["ranking_csv"].split(",") if row["ranking_csv"] else []
            changes = json.loads(row["elo_changes_json"]) if row["elo_changes_json"] else {}

            with st.expander(f"Partie {len(df_sorted)-idx} - {ts.strftime('%d/%m/%Y à %H:%M')}"):
                st.write("**Classement :**")
                for pos, k in enumerate(ranking_keys, 1):
                    name_series = df_players.loc[df_players["key"] == k, "name"]
                    player_name = name_series.iloc[0] if not name_series.empty else k
                    ch = changes.get(k, {})
                    delta = ch.get("delta", 0.0)
                    sign = "+" if delta >= 0 else ""
                    color = "green" if delta >= 0 else "red"

                    col1, col2, col3 = st.columns([1, 3, 2])
                    with col1:
                        st.write(f"**{pos}.**")
                    with col2:
                        st.write(f"**{player_name}**")
                    with col3:
                        st.markdown(f":{color}[{ch.get('old','?')} → {ch.get('new','?')} ({sign}{delta})]")

elif menu == "Statistiques":
    st.header("Statistiques des Joueurs")
    df_players = load_players_df()
    df_matches = load_matches_df()

    if df_players.empty:
        st.info("Aucun joueur enregistré.")
    elif df_matches.empty:
        st.info("Aucune partie jouée. Les statistiques apparaîtront après les premières parties.")
    else:
        # Préparer stats par joueur
        stats = {}
        for _, p in df_players.iterrows():
            stats[p["key"]] = {
                "name": p["name"],
                "current_elo": float(p["elo"]),
                "matches_played": 0,
                "wins": 0,
                "podiums": 0,
                "elo_history": [BASE_ELO],
                "min_elo": BASE_ELO,
                "max_elo": BASE_ELO,
            }

        # Parcourir matches en ordre chronologique
        df_chrono = df_matches.sort_values("timestamp", ascending=True)
        for _, m in df_chrono.iterrows():
            ranking = (m["ranking_csv"].split(",") if m["ranking_csv"] else [])
            players = (m["players_csv"].split(",") if m["players_csv"] else [])
            changes = json.loads(m["elo_changes_json"]) if m["elo_changes_json"] else {}

            for k in players:
                if k not in stats:
                    continue
                stats[k]["matches_played"] += 1
                if ranking and ranking[0] == k:
                    stats[k]["wins"] += 1
                if ranking:
                    pos = ranking.index(k)
                    if pos < 3:
                        stats[k]["podiums"] += 1
                # Historique Elo
                if k in changes and "new" in changes[k]:
                    new_elo = float(changes[k]["new"])
                    stats[k]["elo_history"].append(new_elo)
                    stats[k]["min_elo"] = min(stats[k]["min_elo"], new_elo)
                    stats[k]["max_elo"] = max(stats[k]["max_elo"], new_elo)

        sorted_items = sorted(stats.items(), key=lambda kv: kv[1]["current_elo"], reverse=True)
        for k, s in sorted_items:
            if s["matches_played"] > 0:
                with st.expander(f"**{s['name']}** - {s['current_elo']:.1f} Elo"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Parties jouées", s["matches_played"])
                    with col2:
                        win_rate = (s["wins"] / s["matches_played"] * 100) if s["matches_played"] > 0 else 0
                        st.metric("Victoires", f"{s['wins']} ({win_rate:.0f}%)")
                    with col3:
                        st.metric("Podiums", s["podiums"])
                    with col4:
                        avg_elo = sum(s["elo_history"]) / len(s["elo_history"])
                        st.metric("Elo moyen", f"{avg_elo:.1f}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Elo minimum", f"{s['min_elo']:.1f}")
                    with col2:
                        st.metric("Elo maximum", f"{s['max_elo']:.1f}")

elif menu == "Graphiques":
    st.header("Évolution des Classements Elo")
    df_players = load_players_df()
    df_matches = load_matches_df()

    if df_matches.empty:
        st.info("Aucune partie jouée. Les graphiques apparaîtront après les premières parties.")
    else:
        # Construire séries par joueur
        # Timestamps triés
        df_chrono = df_matches.sort_values("timestamp", ascending=True).reset_index(drop=True)
        first_ts = None
        if not df_chrono.empty:
            try:
                first_ts = datetime.fromisoformat(df_chrono.loc[0, "timestamp"])
            except Exception:
                first_ts = datetime.utcnow()
        player_series = {}
        for _, p in df_players.iterrows():
            player_series[p["key"]] = {
                "name": p["name"],
                "timestamps": [],
                "elos": [BASE_ELO],
            }

        for _, m in df_chrono.iterrows():
            try:
                ts = datetime.fromisoformat(m["timestamp"])
            except Exception:
                ts = datetime.utcnow()
            players = (m["players_csv"].split(",") if m["players_csv"] else [])
            changes = json.loads(m["elo_changes_json"]) if m["elo_changes_json"] else {}
            for k in players:
                if k in player_series and k in changes and "new" in changes[k]:
                    player_series[k]["timestamps"].append(ts)
                    player_series[k]["elos"].append(float(changes[k]["new"]))

        fig = go.Figure()
        for k, data in player_series.items():
            if data["timestamps"]:
                base_point_time = (first_ts - timedelta(seconds=1)) if first_ts else datetime.utcnow()
                xs = [base_point_time] + data["timestamps"]
                ys = data["elos"]
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines+markers',
                    name=data["name"],
                    line=dict(width=2),
                    marker=dict(size=6)
                ))

        fig.update_layout(
            title="Évolution des Elo au fil des parties",
            xaxis_title="Date",
            yaxis_title="Elo",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Focus sur un joueur
        with_names = [data["name"] for data in player_series.values() if len(data["timestamps"]) > 0]
        if with_names:
            selected_player_name = st.selectbox("Voir le détail pour un joueur", with_names)
            # retrouver la clé
            selected_key = None
            for k, data in player_series.items():
                if data["name"] == selected_player_name and data["timestamps"]:
                    selected_key = k
                    break
            if selected_key:
                data = player_series[selected_key]
                base_point_time = (first_ts - timedelta(seconds=1)) if first_ts else datetime.utcnow()
                xs = [base_point_time] + data["timestamps"]
                ys = data["elos"]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines+markers',
                    name=selected_player_name,
                    line=dict(width=3),
                    marker=dict(size=8),
                    fill='tozeroy'
                ))
                fig2.update_layout(
                    title=f"Évolution de {selected_player_name}",
                    xaxis_title="Date",
                    yaxis_title="Elo",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

elif menu == "Exporter":
    st.header("Exporter les Données")
    df_players = load_players_df()
    df_matches = load_matches_df()

    st.subheader("Exporter les joueurs")
    if not df_players.empty:
        # CSV joueurs
        df_sorted = df_players.sort_values("elo", ascending=False)
        csv_players = "Nom,Elo\n" + "\n".join(
            f"{row['name']},{float(row['elo']):.1f}" for _, row in df_sorted.iterrows()
        )
        st.download_button(
            label="📥 Télécharger les joueurs (CSV)",
            data=csv_players,
            file_name="joueurs_elo.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucun joueur à exporter.")

    st.divider()

    st.subheader("Exporter l'historique des parties")
    if not df_matches.empty:
        csv_matches = "Date,Heure,Premier,Deuxieme,Troisieme,Quatrieme\n"
        for _, m in df_matches.iterrows():
            try:
                dt = datetime.fromisoformat(m["timestamp"])
            except Exception:
                dt = datetime.utcnow()
            date_str = dt.strftime("%d/%m/%Y")
            time_str = dt.strftime("%H:%M:%S")

            ranking_keys = m["ranking_csv"].split(",") if m["ranking_csv"] else []
            changes = json.loads(m["elo_changes_json"]) if m["elo_changes_json"] else {}

            players_out = []
            for k in ranking_keys:
                name_series = df_players.loc[df_players["key"] == k, "name"]
                player_name = name_series.iloc[0] if not name_series.empty else k
                delta = changes.get(k, {}).get("delta", 0.0)
                players_out.append(f"{player_name} ({delta:+.1f})")

            while len(players_out) < 4:
                players_out.append("")

            csv_matches += f"{date_str},{time_str},{players_out[0]},{players_out[1]},{players_out[2]},{players_out[3]}\n"

        st.download_button(
            label="📥 Télécharger l'historique (CSV)",
            data=csv_matches,
            file_name="historique_parties.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucune partie à exporter.")

# Sidebar infos
st.sidebar.divider()
st.sidebar.caption("💾 Base de données: Google Sheets (connexion sécurisée)")
try:
    _df_p = load_players_df()
    _df_m = load_matches_df()
    st.sidebar.caption(f"👥 {len(_df_p)} joueurs enregistrés")
    st.sidebar.caption(f"🎯 {len(_df_m)} parties jouées")
except Exception:
    st.sidebar.caption("⚠️ Impossible de charger les compteurs (vérifier la connexion)")
