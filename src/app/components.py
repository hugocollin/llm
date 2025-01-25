"""
Ce fichier contient les fonctions nécessaires pour l'affichage de l'interface de l'application.
"""

import time
import streamlit as st

def stream_text(text: str):
    """
    Fonction pour afficher le texte progressivement.

    Args:
        text (str): Texte à afficher progressivement.
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

def get_new_chat_name() -> str:
    """
    Fonction pour obtenir le nom d'une nouvelle conversation.

    Returns:
        str: Nom de la nouvelle conversation.
    """

    existing_numbers = [
        int(name.split(" ")[1])
        for name in st.session_state["chats"].keys()
        if name.startswith("Chat ") and name.split(" ")[1].isdigit()
    ]
    n = 1
    while n in existing_numbers:
        n += 1
    return f"Chat {n}"

def show_sidebar() -> str:
    """
    Fonction pour afficher la barre latérale de l'application.

    Returns:
        str: Nom de la conversation sélectionnée.
    """

    # Initialisation des conversations
    if "chats" not in st.session_state:
        st.session_state["chats"] = {}

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
                    new_chat_name = get_new_chat_name()
                    st.session_state["chats"][new_chat_name] = []
                    st.session_state["selected_chat"] = new_chat_name
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
                btn_cols = st.columns([3, 1])
                with btn_cols[0]:
                    if st.button(f":material/forum: {chat_name}"):
                        selected_chat = chat_name
                with btn_cols[1]:
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
            chats_to_analyze = {k: v for k, v in st.session_state["chats"].items() if k in selected_conversations}

            # Initialisation des variables pour les statistiques
            total_messages = 0
            total_latency = 0.0
            total_cost = 0.0
            total_energy = 0.0
            total_gwp = 0.0

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

            average_latency = total_latency / (total_messages / 2 or 1)

            # Affichage des statistiques
            cols = st.columns(3)
            with cols[0]:
                with st.container(border=True):
                    st.write("**🗨️ Nombre total de messages envoyés**")
                    st.title(total_messages)
            with cols[1]:
                with st.container(border=True):
                    st.write("**📶 Latence moyenne des réponses**")
                    st.title(f"{average_latency:.2f} secondes")
            with cols[2]:
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
            st.info("Veuillez sélectionner au moins une conversation pour afficher les statistiques.", icon=":material/info:")
    else:
        st.info("Veuillez commencer une conversation pour afficher les statistiques.", icon=":material/info:")
