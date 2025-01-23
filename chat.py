import streamlit as st
import os
from dotenv import find_dotenv, load_dotenv

from components import stream_text

class Chat:
    def __init__(self):
        """
        Initialise la classe Chat avec la clé API nécessaire pour utiliser le modèle Mistral.
        """
        # Récupération de la clé API Mistral
        try:
            load_dotenv(find_dotenv())
            self.API_KEY = os.getenv("MISTRAL_API_KEY")
        except FileNotFoundError:
            self.API_KEY = st.secrets["MISTRAL_API_KEY"]
        
        # Définition du prompt de rôle pour l'IA [TEMP]
        self.role_prompt = ""

    def run(self):
        """
        Lance l'affichage du chat avec l'IA Mistral.
        """
        
        # Avertissement si la clé API Mistral n'est pas présente
        if not self.API_KEY:
            st.error(
                "**Application indisponible :** Vous n'avez pas rajouté votre clé API Mistral dans les fichiers de l'application. "
                "Veuillez ajouter le fichier `.env` à la racine du projet puis redémarrer l'application.", icon="⚠️"
            )
            st.session_state['found_mistral_api'] = False
        else:
            st.session_state['found_mistral_api'] = True

        # Vérification si l'historique de la conversation est initialisé
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Affichage de l'historique de la conversation
        for message in st.session_state.messages:
            # Affichage des messages de l'utilisateur
            if message["role"] == "User":
                with st.chat_message(message["role"], avatar="👤"):
                    st.write(message["content"])
            # Affichage des messages de l'IA
            elif message["role"] == "assistant":
                with st.chat_message(message["role"], avatar="✨"):
                    st.markdown(message["content"])
                    metrics = message["metrics"]
                    st.markdown(
                        f"📶 *Latence : {metrics['latency']:.2f} secondes* | "
                        f"💲 *Coût : {metrics['euro_cost']:.6f} €* | "
                        f"⚡ *Utilisation énergétique : {metrics['energy_usage']} kWh* | "
                        f"🌡️ *Potentiel de réchauffement global : {metrics['gwp']} kgCO2eq*"
                    )

        # Zone de saisie pour le chat avec l'IA [TEMP]
        if message := st.chat_input(
            placeholder="Écrivez votre message", key="search_restaurant_temp", 
            disabled=not st.session_state.get('found_mistral_api', False)
        ):
            if message.strip():

                # Affichage du nouveau message de l'utilisateur
                with st.chat_message("user", avatar="👤"):
                    st.write(message)

                # Ajout du message à l'historique de la conversation
                st.session_state.messages.append({"role": "User", "content": message})

                # Initialisation des connaissances de l'IA
                if 'bdd_chunks' not in st.session_state:
                    with st.spinner("Démarrage de l'IA..."):
                        st.session_state['bdd_chunks'] = instantiate_bdd()

                if 'llm' not in st.session_state:
                    st.session_state['llm'] = AugmentedRAG(
                        role_prompt=self.role_prompt,
                        generation_model="mistral-large-latest",
                        bdd_chunks=st.session_state['bdd_chunks'],
                        top_n=3,
                        max_tokens=3000,
                        temperature=0.3,
                    )

                # Récupération de la réponse de l'IA
                llm = st.session_state['llm']
                response = llm(
                    query=message,
                    history=st.session_state.messages,
                )

                # Affichage de la réponse de l'IA
                with st.chat_message("AI", avatar="✨"):
                    st.write_stream(stream_text(response["response"]))
                    st.markdown(
                        f"📶 *Latence : {response['latency']:.2f} secondes* | "
                        f"💲 *Coût : {response['euro_cost']:.6f} €* | "
                        f"⚡ *Utilisation énergétique : {response['energy_usage']} kWh* | "
                        f"🌡️ *Potentiel de réchauffement global : {response['gwp']} kgCO2eq*"
                    )

                # Ajout de la réponse de l'IA à l'historique de la conversation
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["response"],
                    "metrics": {
                        "latency": response["latency"],
                        "euro_cost": response["euro_cost"],
                        "energy_usage": response["energy_usage"],
                        "gwp": response["gwp"]
                    }
                })