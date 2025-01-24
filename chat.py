import streamlit as st
import os
from dotenv import find_dotenv, load_dotenv

from components import stream_text

class Chat:
    def __init__(self, selected_chat):
        """
        Initialise la classe Chat avec la clé API nécessaire pour utiliser le modèle Mistral.

        Args:
            selected_chat (str): Le chat sélectionné pour la conversation.
        """

        # Récupération de la clé API Mistral
        try:
            load_dotenv(find_dotenv())
            self.API_KEY = os.getenv("MISTRAL_API_KEY")
        except FileNotFoundError:
            self.API_KEY = st.secrets["MISTRAL_API_KEY"]
        
        # Récupération du chat sélectionné
        self.selected_chat = selected_chat

        # Initialisation des messages du chat
        if "chats" not in st.session_state:
            st.session_state["chats"] = {}

        # Vérification si le chat sélectionné existe
        if self.selected_chat not in st.session_state["chats"]:
            st.session_state["chats"][self.selected_chat] = []
        
        # Définition du prompt de rôle pour l'IA [TEMP]
        self.role_prompt = ""

    def run(self):
        """
        Lance l'affichage du chat avec l'IA Mistral.
        """
        
        # Avertissement si la clé API Mistral n'est pas présente
        if not self.API_KEY:
            st.error(
                "**Conversation avec l'IA indisponible :** Votre clé d'API Mistral est introuvable.", icon=":material/error:")
            st.session_state['found_mistral_api'] = False
        else:
            st.session_state['found_mistral_api'] = True

        # Assignation des messages du chat sélectionné
        st.session_state.messages = st.session_state["chats"][self.selected_chat]

        # Mise en page du chat
        st.markdown(
            """
            <style>
            .message-container {
                height: 70vh; /* 70% de la hauteur de la fenêtre */
                overflow-y: auto;
                padding: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Affichage de l'historique de la conversation
        with st.container():
            st.markdown('<div class="message-container">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)

        
        cols = st.columns([3, 10, 1])
        # Choix du modèle [TEMP]
        with cols[0]:
            st.selectbox("", label_visibility="collapsed", options=["Option 1", "Option 2", "Option 3"], index=0)

        # Zone de saisie pour le chat avec l'IA [TEMP]
        with cols[1]:
            if message := st.chat_input(
                placeholder="Écrivez votre message", key=f"chat_input_{self.selected_chat}", 
                disabled=not st.session_state.get('found_mistral_api', False)
            ):
                if message.strip():

                    # Affichage du nouveau message de l'utilisateur
                    with st.chat_message("User", avatar="👤"):
                        st.write(message)

                    # Ajout du message à l'historique de la conversation
                    st.session_state.messages.append({"role": "User", "content": message})

                    # # Initialisation des connaissances de l'IA
                    # if 'bdd_chunks' not in st.session_state:
                    #     with st.spinner("Démarrage de l'IA..."):
                    #         st.session_state['bdd_chunks'] = instantiate_bdd()

                    # if 'llm' not in st.session_state:
                    #     st.session_state['llm'] = AugmentedRAG(
                    #         role_prompt=self.role_prompt,
                    #         generation_model="mistral-large-latest",
                    #         bdd_chunks=st.session_state['bdd_chunks'],
                    #         top_n=3,
                    #         max_tokens=3000,
                    #         temperature=0.3,
                    #     )

                    # # Récupération de la réponse de l'IA
                    # llm = st.session_state['llm']
                    # response = llm(
                    #     query=message,
                    #     history=st.session_state.messages,
                    # )

                    # Affichage d'un faux message temporaire
                    response = {
                        "response": "Je ne suis pas encore prêt à répondre à vos questions, mais je le serai bientôt !",
                        "latency": 0,
                        "euro_cost": 0,
                        "energy_usage": 0,
                        "gwp": 0
                    }

                    # Affichage de la réponse de l'IA
                    with st.chat_message("assistant", avatar="✨"):
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
                            "latency": response['latency'],
                            "euro_cost": response['euro_cost'],
                            "energy_usage": response['energy_usage'],
                            "gwp": response['gwp']
                        }
                    })

        # Bouton pour ajouter un fichier [TEMP]
        with cols[2]:
            if st.button("", icon=":material/attach_file:"):
                st.toast("Fonctionnalité disponible ultérieurement", icon=":material/info:")

        # Sauvegarde des messages dans l'espace de discussion sélectionné
        st.session_state["chats"][self.selected_chat] = st.session_state.messages