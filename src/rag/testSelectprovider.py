import streamlit as st
import json
from typing import Dict, Any
import asyncio
import os
from dotenv.main import load_dotenv
from model_api import MultiModelLLM  # Assurez-vous que le nom du fichier est correct

load_dotenv()

def display_config(config: Dict[str, Any]):
    """Affiche la configuration de manière organisée."""
    st.header("Configuration actuelle")
    
    # Affichage du provider et modèle actuels
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Provider actuel", config["current_provider"])
    with col2:
        st.metric("Modèle actuel", config["current_model"])
    
    # Affichage détaillé des providers et leurs modèles
    st.subheader("Providers disponibles")
    for provider, details in config["providers"].items():
        with st.expander(f"📦 {provider.capitalize()}"):
            st.write("Modèles disponibles:")
            for model in details["models"]:
                st.code(model, language="plain")
    
    # Affichage des capacités
    st.subheader("Capacités")
    for capability in config["capabilities"]:
        st.markdown(f"- ✨ {capability}")
    
    # Affichage JSON brut
    with st.expander("Voir la configuration JSON brute"):
        st.json(config)

def switch_provider_ui(llm: MultiModelLLM, config: Dict[str, Any]):
    """Interface pour changer de provider et de modèle."""
    st.header("Changer de configuration")
    
    # Sélection du provider
    provider = st.selectbox(
        "Choisir un provider",
        options=list(config["providers"].keys()),
        index=list(config["providers"].keys()).index(config["current_provider"])
    )
    
    # Sélection du modèle en fonction du provider
    model = st.selectbox(
        "Choisir un modèle",
        options=config["providers"][provider]["models"]
    )
    
    if st.button("Appliquer les changements"):
        try:
            llm.switch_provider(provider, model)
            st.success(f"Configuration mise à jour avec succès: {provider} - {model}")
            return True
        except ValueError as e:
            st.error(f"Erreur lors du changement: {str(e)}")
            return False
    return False

def initialize_llm():
    """Initialise le LLM avec les clés API."""
    # Récupération des clés API
    mistral_key = os.getenv("MISTRAL_API_KEY", "bjLwvexOwElfrChA7jKgPBu9MEjpd6vD")
    gemini_key = os.getenv("GEMINI_API_KEY", "AIzaSyBo-Ib2DP5cGfYUoKdiqIuR2bU2YLgXZV4")
    
    # Initialisation avec les clés API
    return MultiModelLLM(
        api_key_mistral=mistral_key,
        api_key_gemini=gemini_key,
        default_model="Mistral-Large-Instruct-2411",  # Modèle par défaut
        default_provider="mistral"  # Provider par défaut
    )

def main():
    st.set_page_config(
        page_title="LLM Config Tester",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LLM Configuration Tester")
    
    # Initialisation du LLM
    try:
        llm = initialize_llm()
        st.success("✅ LLM initialisé avec succès")
    except ValueError as e:
        st.error(f"❌ Erreur d'initialisation du LLM: {str(e)}")
        st.stop()
    
    # Container pour la configuration
    with st.container():
        # Récupération de la configuration
        config = llm.get_model_config()
        
        # Affichage de la configuration
        display_config(config)
        
        # Interface de changement de provider
        st.divider()
        if switch_provider_ui(llm, config):
            # Rafraîchissement de la configuration après changement
            st.rerun()

if __name__ == "__main__":
    main()