import streamlit as st

from chat import Chat

# Configuration de la page
st.set_page_config(page_title="LLM", page_icon="✨", layout="wide")

# Affichage du titre personnalisé
st.title('Bonjour [utilisateur] !')

# Création de l'instance du chat
chat = Chat()

# Affichage du chat
chat.run()