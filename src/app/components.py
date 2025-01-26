"""
Ce fichier contient les fonctions nécessaires pour l'affichage de l'interface de l'application.
"""

import os
import time
import streamlit as st
from dotenv import find_dotenv, load_dotenv

def load_api_keys():
    """
    Fonction pour charger et stocker les clés API.
    """
    # Recherche des clés API si l'application est utilisé en local
    try:
        load_dotenv(find_dotenv())
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    # Sinon recherche des clés API si l'application est utilisé en ligne
    except FileNotFoundError:
        mistral_api_key = st.secrets["MISTRAL_API_KEY"]
        gemini_api_key = st.secrets["GEMINI_API_KEY"]

    # Stockage du statut de recherche des clés API
    if mistral_api_key and gemini_api_key:
        st.session_state["found_api_keys"] = True
    else:
        st.session_state["found_api_keys"] = False

def stream_text(text: str):
    """
    Fonction pour afficher le texte progressivement.

    Args:
        text (str): Texte à afficher progressivement.
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

def create_new_chat():
    """
    Fonction pour créer une nouvelle conversation.
    """

    # Récupération des numéros de conversation existants
    existing_numbers = [
        int(name.split(" ")[1])
        for name in st.session_state["chats"].keys()
        if name.startswith("Conversation ") and name.split(" ")[1].isdigit()
    ]

    # Recherche du prochain numéro de conversation disponible
    n = 1
    while n in existing_numbers:
        n += 1

    # Création de la nouvelle conversation
    new_chat_name = f"Conversation {n}"
    st.session_state["chats"][new_chat_name] = []
    st.session_state["selected_chat"] = new_chat_name

@st.dialog("Renommer la conversation")
def rename_chat(current_name: str):
    """
    Fonction pour renommer une conversation.
    """
    
    # Saisie du nouveau nom de la conversation
    new_name = st.text_input(
        "Saisissez le nouveau nom de la conversation :",
        value=current_name,
        max_chars=30
    )

    if st.button("Enregistrer", icon=":material/save_as:"):
        # Vérification de la conformité du nouveau nom
        if new_name in st.session_state["chats"] and new_name != current_name:
            st.error(
                "Ce nom de conversation existe déjà, veuillez en choisir un autre.",
                icon=":material/error:"
            )
        # Enregistrement du nouveau nom
        else:
            st.session_state["chats"][new_name] = st.session_state["chats"].pop(current_name)
            st.session_state["selected_chat"] = new_name
            if new_name != current_name:
                st.session_state["chat_renamed"] = True
            st.rerun()

def show_sidebar() -> str:
    """
    Fonction pour afficher la barre latérale de l'application.

    Returns:
        str: Nom de la conversation sélectionnée.
    """

    # Initialisation des conversations
    if "chats" not in st.session_state:
        st.session_state["chats"] = {}

    # Initialisation de la variable d'état de renommage de conversation
    if "chat_renamed" not in st.session_state:
        st.session_state["chat_renamed"] = False

    with st.sidebar:
        # Titre de l'application
        st.title("✨ SISE Classmate")

        # Auteurs
        st.write(
            "*Cette application a été développée par "
            "[KPAMEGAN Falonne](https://github.com/marinaKpamegan), "
            "[KARAMOKO Awa](https://github.com/karamoko17), "
            "[CISSE Lansana](https://github.com/lansanacisse) "
            "et [COLLIN Hugo](https://github.com/hugocollin), dans le cadre du Master 2 SISE.*"
        )

        header_cols = st.columns([3, 1, 1])

        # Section des conversations
        with header_cols[0]:
            st.header("Conversations")

        # Bouton pour afficher les statistiques
        with header_cols[1]:
            st.write("")
            if st.button("", icon=":material/bar_chart:"):
                show_stats_dialog()

        # Bouton pour ajouter un chat
        with header_cols[2]:
            st.write("")
            if st.button("", icon=":material/add_comment:"):
                if len(st.session_state["chats"]) < 5:
                    create_new_chat()
                else:
                    st.toast(
                        "Nombre maximal de conversations atteint, "
                        "supprimez-en une pour en commencer une nouvelle",
                        icon=":material/feedback:",
                    )

        # Sélecteur d'espaces de discussion
        if st.session_state["chats"]:
            selected_chat = None
            for chat_name in list(st.session_state["chats"].keys()):
                btn_cols = st.columns([3, 1, 1])

                # Bouton pour sélectionner le chat
                with btn_cols[0]:
                    if st.button(f":material/forum: {chat_name}"):
                        selected_chat = chat_name

                # Boutons pour renommer le chat
                with btn_cols[1]:
                    if st.button(
                        "", icon=":material/edit:", key=f"rename_'{chat_name}'_button"
                    ):
                        rename_chat(chat_name)
                    if st.session_state["chat_renamed"] is True:
                        st.toast(
                            "Conversation renommée avec succès !",
                            icon=":material/check_circle:"
                        )
                        st.session_state["chat_renamed"] = False

                # Bouton pour supprimer le chat
                with btn_cols[2]:
                    if st.button(
                        "", icon=":material/delete:", key=f"delete_'{chat_name}'_button"
                    ):
                        del st.session_state["chats"][chat_name]
                        del st.session_state[f"delete_'{chat_name}'_button"]
                        if st.session_state.get("selected_chat") == chat_name:
                            st.session_state["selected_chat"] = next(
                                iter(st.session_state["chats"]), None
                            )
                        st.rerun()
            return selected_chat
        else:
            # Message d'information si aucune conversation n'a été créée
            st.info(
                "Pour commencer une nouvelle conversation, "
                "cliquez sur le bouton :material/add_comment:",
                icon=":material/info:",
            )
            return None

@st.dialog("Statistiques de conversation", width="large")
def show_stats_dialog():
    """
    Fonction pour afficher les statistiques globales ou détaillées par conversation.
    """
    # Récupération des noms des conversations
    conversations = list(st.session_state.get("chats", {}).keys())

    if conversations:
        # Sélection des conversations à analyser
        selected_conversations = st.pills(
            label="Sélectionnez les conversations à analyser :",
            options=conversations,
            selection_mode="multi",
            default=conversations
        )

        # Récupération des conversations à analyser
        if selected_conversations:
            chats_to_analyze = {
                k: v
                for k, v in st.session_state["chats"].items()
                if k in selected_conversations
            }

            # Initialisation des variables pour les statistiques
            total_messages = 0
            total_latency = 0.0
            total_cost = 0.0
            total_energy = 0.0
            total_gwp = 0.0
            total_blocked_messages = 0

            # Calcul des statistiques
            for chat in chats_to_analyze.values():
                for message in chat:
                    total_messages += 1
                    if message["role"] == "AI":
                        metrics = message.get("metrics", {})
                        total_latency += metrics.get("latency", 0.0)
                        total_cost += metrics.get("euro_cost", 0.0)
                        total_energy += metrics.get("energy_usage", 0.0)
                        total_gwp += metrics.get("gwp", 0.0)
                    if message["role"] == "Guardian":
                        total_blocked_messages += 1

            sent_messages = total_messages / 2
            average_latency = total_latency / (total_messages / 2 or 1)

            # Affichage des statistiques
            cols = st.columns(4)
            with cols[0]:
                with st.container(border=True):
                    st.write("**🗨️ Nombre total de messages envoyés**")
                    st.title(f"{sent_messages:.0f}")
            with cols[1]:
                with st.container(border=True):
                    st.write("**🛡️ Nombre total de messages bloqués**")
                    st.title(f"{total_blocked_messages:.0f}")
            with cols[2]:
                with st.container(border=True):
                    st.write("**📶 Latence moyenne des réponses**")
                    st.title(f"{average_latency:.2f} secondes")
            with cols[3]:
                with st.container(border=True):
                    st.write("**💲 Coût total**")
                    st.title(f"{total_cost:.2f} €")
            cols = st.columns(2)
            with cols[0]:
                with st.container(border=True):
                    st.write("**⚡ Utilisation énergétique totale**")
                    st.title(f"{total_energy:.2f} kWh")
            with cols[1]:
                with st.container(border=True):
                    st.write("**🌡️ Potentiel de réchauffement global total**")
                    st.title(f"{total_gwp:.2f} kgCO2eq")
        else:
            # Message d'information si aucune conversation n'a été sélectionnée
            st.info(
                "Veuillez sélectionner au moins une conversation pour afficher les statistiques.",
                icon=":material/info:",
            )
    else:
        # Message d'information si aucune conversation n'a été créée
        st.info(
            "Veuillez commencer une conversation pour afficher les statistiques.",
            icon=":material/info:",
        )
