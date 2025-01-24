import streamlit as st

from chat import Chat
from components import show_sidebar

# Configuration de la page
st.set_page_config(page_title="LLM", page_icon="✨", layout="wide")

# Affichage de la barre latérale
selected_chat = show_sidebar()

# Stockage du chat sélectionné
if selected_chat:
    st.session_state['selected_chat'] = selected_chat
elif 'selected_chat' not in st.session_state and selected_chat:
    st.session_state['selected_chat'] = selected_chat

# Affichage du chat sélectionné
if 'selected_chat' in st.session_state:
    current_chat = st.session_state['selected_chat']
    st.subheader(f"{current_chat}")
    
    # Initialisation de l'historique des messages pour le chat sélectionné
    if current_chat not in st.session_state["chats"]:
        st.session_state["chats"][current_chat] = []
    
    # Création de l'instance du chat en passant le chat sélectionné
    chat = Chat(selected_chat=current_chat)
    
    # Affichage du chat sélectionné
    chat.run()
    
    # Sauvegarde des messages du chat sélectionné
    st.session_state["chats"][current_chat] = st.session_state.get("chats", {}).get(current_chat, [])