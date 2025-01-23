import streamlit as st

from chat import Chat
from components import show_sidebar

# Configuration de la page
st.set_page_config(page_title="LLM", page_icon="✨", layout="wide")
st.title("Bonjour [utilisateur] !")

# Affichage de la barre latérale
selected_chat = show_sidebar()

# Affichage du chat sélectionné
if selected_chat:
    st.subheader(f"{selected_chat}")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Récupération des messages du chat sélectionné
    st.session_state["messages"] = st.session_state["chats"][selected_chat]

    # Création de l'instance du chat
    chat = Chat()

    # Affichage du chat sélectionné
    chat.run()
    
    # Sauvegarde des messages du chat sélectionné
    st.session_state["chats"][selected_chat] = st.session_state["messages"]