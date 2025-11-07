import streamlit as st
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

DB_PATH = Path("elo_db.json")
BASE_ELO = 1000.0
K_FACTOR = 32.0

def load_db():
    if DB_PATH.exists():
        with DB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if "players" not in data:
                data = {"players": data, "matches": []}
            if "matches" not in data:
                data["matches"] = []
            return data
    return {"players": {}, "matches": []}

def save_db(db):
    with DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def key_name(name: str) -> str:
    return name.strip().lower()

def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

def ensure_players(db, names):
    keys = []
    for name in names:
        k = key_name(name)
        if k not in db["players"]:
            db["players"][k] = {"name": name.strip(), "elo": BASE_ELO}
        else:
            db["players"][k]["name"] = name.strip()
        keys.append(k)
    return keys

def update_ratings_multiplayer(db, sel_keys, ranking_keys, K=K_FACTOR):
    pos = {k: i for i, k in enumerate(ranking_keys)}
    n = len(sel_keys)
    olds = {k: db["players"][k]["elo"] for k in sel_keys}
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
    for k in sel_keys:
        db["players"][k]["elo"] = news[k]
    return olds, news

def record_match(db, sel_keys, ranking_keys, olds, news):
    match_record = {
        "timestamp": datetime.now().isoformat(),
        "players": sel_keys,
        "ranking": ranking_keys,
        "elo_changes": {}
    }
    for k in sel_keys:
        match_record["elo_changes"][k] = {
            "old": olds[k],
            "new": news[k],
            "delta": news[k] - olds[k]
        }
    db["matches"].append(match_record)
    return match_record

st.set_page_config(
    page_title="Classement Elo Fléchettes",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Classement Elo Fléchettes")

db = load_db()

menu = st.sidebar.radio(
    "Navigation",
    ["Classement", "Nouvelle Partie", "Ajouter un Joueur", "Historique", "Statistiques", "Graphiques", "Exporter"]
)

if menu == "Classement":
    st.header("Classement des Joueurs")
    
    if not db["players"]:
        st.info("Aucun joueur enregistré. Ajoutez des joueurs pour commencer !")
    else:
        players_list = []
        for key, data in db["players"].items():
            players_list.append({
                "Nom": data["name"],
                "Elo": round(data["elo"], 1),
                "key": key
            })
        
        players_list.sort(key=lambda x: x["Elo"], reverse=True)
        
        for idx, player in enumerate(players_list, 1):
            col1, col2, col3 = st.columns([1, 4, 2])
            with col1:
                if idx == 1:
                    st.markdown("🥇")
                elif idx == 2:
                    st.markdown("🥈")
                elif idx == 3:
                    st.markdown("🥉")
                else:
                    st.markdown(f"**{idx}**")
            with col2:
                st.markdown(f"**{player['Nom']}**")
            with col3:
                st.markdown(f"**{player['Elo']}** points")
            st.divider()

elif menu == "Nouvelle Partie":
    st.header("Enregistrer une Nouvelle Partie")
    
    if not db["players"] or len(db["players"]) < 2:
        st.warning("Vous devez avoir au moins 2 joueurs pour enregistrer une partie.")
    else:
        player_names = [data["name"] for data in db["players"].values()]
        
        st.subheader("1. Sélectionnez les joueurs")
        num_players = st.radio("Nombre de joueurs", [2, 3, 4], horizontal=True)
        
        selected_players = []
        cols = st.columns(num_players)
        for i in range(num_players):
            with cols[i]:
                player = st.selectbox(
                    f"Joueur {i+1}",
                    player_names,
                    key=f"player_{i}"
                )
                selected_players.append(player)
        
        if len(selected_players) != len(set(selected_players)):
            st.error("Chaque joueur ne peut être sélectionné qu'une seule fois !")
        else:
            st.subheader("2. Définissez le classement")
            st.write("Indiquez la position de chaque joueur (1 = premier, " + str(num_players) + " = dernier)")
            
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
                    sel_keys = ensure_players(db, selected_players)
                    
                    sorted_players = sorted(selected_players, key=lambda p: rankings[p])
                    ranking_keys = [key_name(p) for p in sorted_players]
                    
                    olds, news = update_ratings_multiplayer(db, sel_keys, ranking_keys, K_FACTOR)
                    
                    record_match(db, sel_keys, ranking_keys, olds, news)
                    
                    save_db(db)
                    
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
    
    new_player_name = st.text_input("Nom du joueur")
    
    if st.button("Ajouter", type="primary"):
        if not new_player_name.strip():
            st.error("Le nom du joueur ne peut pas être vide !")
        else:
            key = key_name(new_player_name)
            if key in db["players"]:
                st.warning(f"Le joueur '{new_player_name}' existe déjà dans la base !")
            else:
                db["players"][key] = {"name": new_player_name.strip(), "elo": BASE_ELO}
                save_db(db)
                st.success(f"✅ Joueur '{new_player_name}' ajouté avec un Elo de {BASE_ELO} !")
                st.rerun()

elif menu == "Historique":
    st.header("Historique des Parties")
    
    if not db["matches"]:
        st.info("Aucune partie enregistrée pour le moment.")
    else:
        st.write(f"**{len(db['matches'])}** parties jouées au total")
        
        if st.button("🗑️ Annuler la dernière partie", type="secondary"):
            if db["matches"]:
                last_match = db["matches"].pop()
                
                for player_key, changes in last_match["elo_changes"].items():
                    if player_key in db["players"]:
                        db["players"][player_key]["elo"] = changes["old"]
                
                save_db(db)
                st.success("✅ Dernière partie annulée avec succès !")
                st.rerun()
        
        st.divider()
        
        for idx, match in enumerate(reversed(db["matches"]), 1):
            with st.expander(f"Partie {len(db['matches']) - idx + 1} - {datetime.fromisoformat(match['timestamp']).strftime('%d/%m/%Y à %H:%M')}"):
                st.write("**Classement :**")
                for pos, player_key in enumerate(match["ranking"], 1):
                    player_name = db["players"].get(player_key, {}).get("name", player_key)
                    changes = match["elo_changes"][player_key]
                    delta = changes["delta"]
                    sign = "+" if delta >= 0 else ""
                    color = "green" if delta >= 0 else "red"
                    
                    col1, col2, col3 = st.columns([1, 3, 2])
                    with col1:
                        st.write(f"**{pos}.**")
                    with col2:
                        st.write(f"**{player_name}**")
                    with col3:
                        st.markdown(f":{color}[{changes['old']} → {changes['new']} ({sign}{delta})]")

elif menu == "Statistiques":
    st.header("Statistiques des Joueurs")
    
    if not db["players"]:
        st.info("Aucun joueur enregistré.")
    elif not db["matches"]:
        st.info("Aucune partie jouée. Les statistiques apparaîtront après les premières parties.")
    else:
        player_stats = {}
        
        for player_key, player_data in db["players"].items():
            player_stats[player_key] = {
                "name": player_data["name"],
                "current_elo": player_data["elo"],
                "matches_played": 0,
                "wins": 0,
                "podiums": 0,
                "elo_history": [BASE_ELO],
                "min_elo": BASE_ELO,
                "max_elo": BASE_ELO
            }
        
        for match in db["matches"]:
            ranking = match["ranking"]
            for player_key in match["players"]:
                if player_key in player_stats:
                    player_stats[player_key]["matches_played"] += 1
                    
                    if ranking[0] == player_key:
                        player_stats[player_key]["wins"] += 1
                    
                    if ranking.index(player_key) < 3:
                        player_stats[player_key]["podiums"] += 1
                    
                    new_elo = match["elo_changes"][player_key]["new"]
                    player_stats[player_key]["elo_history"].append(new_elo)
                    player_stats[player_key]["min_elo"] = min(player_stats[player_key]["min_elo"], new_elo)
                    player_stats[player_key]["max_elo"] = max(player_stats[player_key]["max_elo"], new_elo)
        
        sorted_stats = sorted(player_stats.items(), key=lambda x: x[1]["current_elo"], reverse=True)
        
        for player_key, stats in sorted_stats:
            if stats["matches_played"] > 0:
                with st.expander(f"**{stats['name']}** - {stats['current_elo']:.1f} Elo"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Parties jouées", stats["matches_played"])
                    with col2:
                        win_rate = (stats["wins"] / stats["matches_played"] * 100) if stats["matches_played"] > 0 else 0
                        st.metric("Victoires", f"{stats['wins']} ({win_rate:.0f}%)")
                    with col3:
                        st.metric("Podiums", stats["podiums"])
                    with col4:
                        avg_elo = sum(stats["elo_history"]) / len(stats["elo_history"])
                        st.metric("Elo moyen", f"{avg_elo:.1f}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Elo minimum", f"{stats['min_elo']:.1f}")
                    with col2:
                        st.metric("Elo maximum", f"{stats['max_elo']:.1f}")

elif menu == "Graphiques":
    st.header("Évolution des Classements Elo")
    
    if not db["matches"]:
        st.info("Aucune partie jouée. Les graphiques apparaîtront après les premières parties.")
    else:
        player_elo_evolution = {}
        
        for player_key, player_data in db["players"].items():
            player_elo_evolution[player_key] = {
                "name": player_data["name"],
                "timestamps": [],
                "elos": [BASE_ELO]
            }
        
        for match in db["matches"]:
            timestamp = datetime.fromisoformat(match["timestamp"])
            for player_key in match["players"]:
                if player_key in player_elo_evolution:
                    player_elo_evolution[player_key]["timestamps"].append(timestamp)
                    player_elo_evolution[player_key]["elos"].append(match["elo_changes"][player_key]["new"])
        
        fig = go.Figure()
        
        for player_key, data in player_elo_evolution.items():
            if data["timestamps"]:
                full_timestamps = [datetime.fromisoformat(db["matches"][0]["timestamp"]) - timedelta(seconds=1)] + data["timestamps"]
                fig.add_trace(go.Scatter(
                    x=full_timestamps,
                    y=data["elos"],
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
        
        player_names = [data["name"] for data in db["players"].values() if data["name"] in [d["name"] for d in player_elo_evolution.values() if d["timestamps"]]]
        
        if player_names:
            selected_player_name = st.selectbox("Voir le détail pour un joueur", player_names)
            
            selected_key = None
            for key, data in player_elo_evolution.items():
                if data["name"] == selected_player_name:
                    selected_key = key
                    break
            
            if selected_key and player_elo_evolution[selected_key]["timestamps"]:
                data = player_elo_evolution[selected_key]
                fig2 = go.Figure()
                full_timestamps = [datetime.fromisoformat(db["matches"][0]["timestamp"]) - timedelta(seconds=1)] + data["timestamps"]
                fig2.add_trace(go.Scatter(
                    x=full_timestamps,
                    y=data["elos"],
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
    
    st.subheader("Exporter les joueurs")
    if db["players"]:
        import io
        
        csv_players = "Nom,Elo\n"
        for player_key, player_data in sorted(db["players"].items(), key=lambda x: x[1]["elo"], reverse=True):
            csv_players += f"{player_data['name']},{player_data['elo']:.1f}\n"
        
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
    if db["matches"]:
        csv_matches = "Date,Heure,Premier,Deuxieme,Troisieme,Quatrieme\n"
        for match in db["matches"]:
            dt = datetime.fromisoformat(match["timestamp"])
            date_str = dt.strftime("%d/%m/%Y")
            time_str = dt.strftime("%H:%M:%S")
            
            players = []
            for player_key in match["ranking"]:
                player_name = db["players"].get(player_key, {}).get("name", player_key)
                delta = match["elo_changes"][player_key]["delta"]
                players.append(f"{player_name} ({delta:+.1f})")
            
            while len(players) < 4:
                players.append("")
            
            csv_matches += f"{date_str},{time_str},{players[0]},{players[1]},{players[2]},{players[3]}\n"
        
        st.download_button(
            label="📥 Télécharger l'historique (CSV)",
            data=csv_matches,
            file_name="historique_parties.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucune partie à exporter.")

st.sidebar.divider()
st.sidebar.caption(f"💾 Base de données: {DB_PATH.resolve()}")
st.sidebar.caption(f"👥 {len(db['players'])} joueurs enregistrés")
st.sidebar.caption(f"🎯 {len(db['matches'])} parties jouées")
