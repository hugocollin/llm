import streamlit as st
import os
from dotenv import find_dotenv, load_dotenv

from src.app.chat import Chat
from src.app.components import show_sidebar

# Configuration de la page
st.set_page_config(page_title="SISE Classmate", page_icon="✨", layout="wide")

# Récupération de la clé API Mistral
try:
    load_dotenv(find_dotenv())
    API_KEY = os.getenv("MISTRAL_API_KEY")
except FileNotFoundError:
    API_KEY = st.secrets["MISTRAL_API_KEY"]

if API_KEY:
    st.session_state['found_mistral_api'] = True
else:
    st.session_state['found_mistral_api'] = False

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
    
    # Création de l'instance du chat en passant le chat sélectionné et une question initiale si disponible
    initial_question = st.session_state.get('initial_question', None)
    chat = Chat(selected_chat=current_chat, initial_question=initial_question)
    
    # Affichage du chat sélectionné
    chat.run()
    
    # Sauvegarde des messages du chat sélectionné
    st.session_state["chats"][current_chat] = st.session_state.get("chats", {}).get(current_chat, [])
else:
    st.container(height=200, border=False)
    with st.container():
        # Affichage si la clé API Mistral n'est pas présente
        if st.session_state['found_mistral_api'] == False:
            # Titre
            st.title("Je ne peux pas vous aider... 😢")

            # Message d'erreur
            st.error("**Conversation avec l'IA indisponible :** Votre clé d'API Mistral est introuvable.", icon=":material/error:")
        else:
            # Titre
            st.title("Comment puis-je vous aider ? 🤩")

            # Barre de saisie de question
            question = st.chat_input("Écrivez votre message", key="new_chat_question")
            
            if question:
                st.session_state['initial_question'] = question
                new_chat = f"Chat {len(st.session_state['chats']) + 1}"
                st.session_state['selected_chat'] = new_chat
                st.session_state['chats'][new_chat] = []
                st.rerun()

            # Suggestions de questions
            suggestion = st.pills(label="NULL", options=["Suggestion question 1", "Suggestion question 2", "Suggestion question 3"], label_visibility="collapsed")
            
            if suggestion:
                st.session_state['initial_question'] = suggestion
                new_chat = f"Chat {len(st.session_state['chats']) + 1}"
                st.session_state['selected_chat'] = new_chat
                st.session_state['chats'][new_chat] = []
                st.rerun()