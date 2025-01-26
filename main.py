"""
Ce fichier contient le code principal de l'application Streamlit.
"""

import os
import streamlit as st
from dotenv import find_dotenv, load_dotenv

from src.app.chat import Chat
from src.app.components import show_sidebar

# Configuration de la page
st.set_page_config(page_title="SISE Classmate", page_icon="✨", layout="wide")

st.write(st.session_state)

# Récupération des clés d'API
try:
    load_dotenv(find_dotenv())
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
except FileNotFoundError:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

if MISTRAL_API_KEY and GEMINI_API_KEY:
    st.session_state["found_api_keys"] = True
else:
    st.session_state["found_api_keys"] = False

# Mise en page personnalisée
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 50px;
            padding-bottom: 0px;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Affichage de la barre latérale
selected_chat = show_sidebar()

# Stockage du chat sélectionné
if selected_chat:
    st.session_state["selected_chat"] = selected_chat
elif "selected_chat" not in st.session_state and selected_chat:
    st.session_state["selected_chat"] = selected_chat

# Affichage du chat sélectionné
if (
    "selected_chat" in st.session_state
    and st.session_state["selected_chat"] is not None
):
    current_chat = st.session_state["selected_chat"]
    st.subheader(f"{current_chat}")

    # Initialisation de l'historique des messages pour le chat sélectionné
    if current_chat not in st.session_state["chats"]:
        st.session_state["chats"][current_chat] = []

    # Création de l'instance du chat
    initial_question = st.session_state.get("initial_question", None)
    chat = Chat(selected_chat=current_chat, initial_question=initial_question)

    # Affichage du chat sélectionné
    chat.run()

    # Sauvegarde des messages du chat sélectionné
    st.session_state["chats"][current_chat] = st.session_state.get("chats", {}).get(
        current_chat, []
    )
else:
    st.container(height=200, border=False)
    with st.container():
        # Affichage si une ou plusieurs clés d'API sont introuvables
        if st.session_state["found_api_keys"] is False:
            # Titre
            st.title("Je ne peux pas vous aider... 😢")

            # Message d'erreur
            st.error(
                "**Conversation avec l'IA indisponible :** "
                "Une ou plusieurs clés d'API sont introuvables.",
                icon=":material/error:",
            )
        else:
            # Titre
            st.title("Comment puis-je vous aider ? 🤩")

            # Barre de saisie de question
            question = st.chat_input("Écrivez votre message", key="new_chat_question")

            if question:
                st.session_state["initial_question"] = question
                new_chat = f"Conversation {len(st.session_state['chats']) + 1}"
                st.session_state["selected_chat"] = new_chat
                st.session_state["chats"][new_chat] = []
                st.rerun()

            # Génération de questions suggérées
            if "suggested_questions" not in st.session_state:
                chat_instance = Chat(selected_chat="suggestions")
                st.session_state["suggested_questions"] = chat_instance.get_suggested_questions()

            # Affichage des suggestions de questions dynamiques
            suggestions = st.pills(
                label=(
                    "Sinon voici quelques suggestions de questions que j'ai générées "
                    "et que vous pouvez me poser :"
                ),
                options=st.session_state["suggested_questions"]
            )

            if suggestions:
                st.session_state["initial_question"] = suggestions
                new_chat = f"Conversation {len(st.session_state['chats']) + 1}"
                st.session_state["selected_chat"] = new_chat
                st.session_state["chats"][new_chat] = []
                st.rerun()
