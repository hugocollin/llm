import streamlit as st

from src.app.chat import Chat
from src.app.components import show_sidebar

# Configuration de la page
st.set_page_config(page_title="SISE Classmate", page_icon="✨", layout="wide")

# Mise en page personnalisée
st.markdown("""
    <style>
        .block-container {
            padding-top: 50px;
            padding-bottom: 0px;
        }
    </style>
""", unsafe_allow_html=True)

# Affichage de la barre latérale
selected_chat = show_sidebar()

# Stockage du chat sélectionné
if selected_chat:
    st.session_state['selected_chat'] = selected_chat
elif 'selected_chat' not in st.session_state and selected_chat:
    st.session_state['selected_chat'] = selected_chat

# Affichage du chat sélectionné
if 'selected_chat' in st.session_state and st.session_state['selected_chat'] is not None:
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
else:
    st.container(height=200, border=False)
    with st.container():
        # Question
        st.title("Comment puis-je vous aider ?")

        # Barre de saisie de question
        question = st.chat_input("Écrivez votre message", key="new_chat_question")
        
        # Suggestions de questions
        st.pills(label="NULL", options=["Suggestion question 1", "Suggestion question 2", "Suggestion question 3"], label_visibility="collapsed")