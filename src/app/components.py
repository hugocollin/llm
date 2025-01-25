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
                st.toast(
                    "Fonctionnalité disponible ultérieurement", icon=":material/info:"
                )

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
