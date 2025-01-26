"""
Ce fichier contient le code principal de l'application Streamlit.
"""

import streamlit as st

from src.app.chat import Chat
from src.app.components import load_api_keys, create_new_chat, show_sidebar

# Configuration de la page
st.set_page_config(page_title="SISE Classmate", page_icon="✨", layout="wide")

# Chargement des clés API
load_api_keys()

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
                create_new_chat()
                st.rerun()

            # Génération de questions suggérées
            if "suggested_questions" not in st.session_state:
                chat_instance = Chat(selected_chat="suggestions")
                st.session_state["suggested_questions"] = chat_instance.get_suggested_questions()

            # Suggestions de questions dynamiques
            suggestions = st.pills(
                label=(
                    "Sinon voici quelques suggestions de questions que j'ai générées "
                    "et que vous pouvez me poser :"
                ),
                options=st.session_state["suggested_questions"]
            )

            if suggestions:
                st.session_state["initial_question"] = suggestions
                create_new_chat()
                st.rerun()
