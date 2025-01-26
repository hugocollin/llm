"""
Ce fichier définit la classe Chat pour gérer les interractions avec l'IA.
"""

import os
import asyncio
import streamlit as st
import PyPDF2

from src.app.components import stream_text
from src.pipeline import EnhancedLLMSecurityManager
from src.rag.model_api import MultiModelLLM

class Chat:
    """
    Classe pour gérer les interractions avec l'IA.
    """

    def __init__(self, selected_chat: str, initial_question: str = None):
        """
        Initialise la classe Chat avec le chat sélectionné et la question initiale si disponible.

        Args:
            selected_chat (str): Chat sélectionné pour la conversation.
            initial_question (str, optionnel): Question initiale à poser à l'IA. None par défaut.
        """

        # Récupération du chat sélectionné
        self.selected_chat = selected_chat

        # Si ce ne sont pas les suggestions
        if selected_chat != "suggestions":
            # Initialisation des messages du chat
            if "chats" not in st.session_state:
                st.session_state["chats"] = {}

            # Vérification si le chat sélectionné existe
            if self.selected_chat not in st.session_state["chats"]:
                st.session_state["chats"][self.selected_chat] = []

            # Stockage de la question initiale
            self.initial_question = initial_question
            st.session_state["initial_question"] = None

            # Mise en page du chat avec l'IA
            self.header_container = st.container()
            self.chat_container = self.header_container.container(height=500)

        # Si les clés d'API sont trouvées
        if st.session_state["found_api_keys"] is True:
            # Initialisation du LLM
            if "llm" not in st.session_state:
                st.session_state["llm"] = MultiModelLLM(
                    api_key_mistral=os.getenv("MISTRAL_API_KEY"),
                    api_key_gemini=os.getenv("GEMINI_API_KEY")
                )
            self.llm = st.session_state["llm"]
        # Si les clés d'API ne sont pas trouvées
        else:
            with self.chat_container.chat_message("", avatar="⚠️"):
                st.write(
                    "**Conversation avec l'IA indisponible :** "
                    "Une ou plusieurs clés d'API sont introuvables."
                )

    def get_suggested_questions(self) -> list:
        """
        Génère 5 exemples de question avec l'IA.

        Returns:
        """
        prompt = (
            "Tu es une intelligence artificielle spécialisée dans l'aide aux élèves à l'école. "
            "Génère 5 questions courtes dans différentes matières sans les préciser, "
            "qu'un élève pourrait te poser sur une notion de cours. "
            "Répond uniquement en donnant les 5 questions sous forme de liste de tirets, "
            "sans explication supplémentaire."
        )
        response = asyncio.run(self.llm.generate(prompt=prompt, model="mistral-large-latest"))
        questions = response["response"].split('\n')
        return [q.strip("- ").strip() for q in questions[:5]]

    def generate_chat_name(self, initial_message: str):
        """
        Génère un nom pour la conversation en fonction du premier message avec l'IA.

        Args:
            initial_message (str): Message initial de la conversation.
        """
        for _ in range(5):
            prompt = (
                "Tu es une intelligence artificielle spécialisée "
                "dans la création de nom de conversation. "
                "En te basant sur le texte suivant, qui est le premier message de la conversation, "
                "propose un nom d'un maximum de 30 caractères pour cette conversation. "
                "Répond uniquement en donnant le nom de la conversation "
                "sans explication supplémentaire. "
                f"Voici le texte : {initial_message}"
            )
            response = asyncio.run(self.llm.generate(prompt=prompt, model="mistral-large-latest"))
            generated_name = response["response"].strip()

            if len(generated_name) > 30:
                continue
            if generated_name in st.session_state["chats"]:
                continue

            st.session_state["chats"][generated_name] = st.session_state["chats"].pop(
                self.selected_chat
            )
            st.session_state["selected_chat"] = generated_name
            self.selected_chat = generated_name
            st.rerun()

    def run(self):
        """
        Lance l'affichage du chat avec l'IA.
        """

        # Affichage de l'historique de la conversation
        for message in st.session_state["chats"][self.selected_chat]:
            # Affichage des messages de l'utilisateur
            if message["role"] == "User":
                with self.chat_container.chat_message(message["role"], avatar="👤"):
                    st.write(message["content"])

            # Affichage des messages de l'IA
            elif message["role"] == "AI":
                with self.chat_container.chat_message(message["role"], avatar="✨"):
                    st.markdown(message["content"])
                    metrics = message["metrics"]
                    st.markdown(
                        f"📶 *Latence : {metrics['latency']:.2f} secondes* | "
                        f"💲 *Coût : {metrics['euro_cost']:.6f} €* | "
                        f"⚡ *Utilisation énergétique : {metrics['energy_usage']} kWh* | "
                        f"🌡️ *Potentiel de réchauffement global : {metrics['gwp']} kgCO2eq*"
                    )

            # Affichage des messages de sécurité
            elif message["role"] == "Guardian":
                with self.chat_container.chat_message(message["role"], avatar="⛔"):
                    st.write(message["content"])

        # Si une question initiale est présente, l'envoyer automatiquement
        if self.initial_question and not st.session_state["chats"][self.selected_chat]:
            self.handle_user_message(self.initial_question)

        # Mise en page de l'interraction avec l'IA
        cols = self.header_container.columns([3, 10, 1])

        # Choix du modèle [TEMP]
        with cols[0]:
            st.selectbox(
                "NULL",
                label_visibility="collapsed",
                options=["Option 1", "Option 2", "Option 3"],
                index=0,
                disabled=not st.session_state.get("found_api_keys", False),
            )

        # Zone de saisie pour le chat avec l'IA
        with cols[1]:
            if message := st.chat_input(
                placeholder="Écrivez votre message",
                key=f"chat_input_{self.selected_chat}",
                disabled=not st.session_state.get("found_api_keys", False),
            ):
                if message.strip():
                    self.handle_user_message(message)

        # Bouton pour ajouter un fichier PDF [TEMP]
        with cols[2]:
            if st.button(
                "",
                icon=":material/attach_file:",
                disabled=not st.session_state.get("found_api_keys", False),
            ):
                self.upload_files_dialog()

    def handle_user_message(self, message: str):
        """
        Gère le message de l'utilisateur et envoie une requête à l'IA.

        Args:
            message (str): Message de l'utilisateur.
        """

        # Affichage du nouveau message de l'utilisateur
        with self.chat_container.chat_message("User", avatar="👤"):
            st.write(message)

        # Ajout du message à l'historique de la conversation
        st.session_state["chats"][self.selected_chat].append(
            {"role": "User", "content": message}
        )

        # Initialisation du pipeline de sécurité
        security_manager = EnhancedLLMSecurityManager(message)

        # Validation du message de l'utilisateur
        is_valid_message = security_manager.validate_input()

        # Si le message de utilisateur est autorisé
        if is_valid_message is True:
            # Envoi du message et récupération de la réponse de l'IA
            response = asyncio.run(self.llm.generate(prompt=message, model="mistral-large-latest"))

            # Affichage de la réponse de l'IA
            with self.chat_container.chat_message("AI", avatar="✨"):
                st.write_stream(stream_text(response["response"]))
                st.markdown(
                    f"📶 *Latence : {response['latency']:.2f} secondes* | "
                    f"💲 *Coût : {response['euro_cost']:.6f} €* | "
                    f"⚡ *Utilisation énergétique : {response['energy_usage']} kWh* | "
                    f"🌡️ *Potentiel de réchauffement global : {response['gwp']} kgCO2eq*"
                )

            # Ajout de la réponse de l'IA à l'historique de la conversation
            st.session_state["chats"][self.selected_chat].append(
                {
                    "role": "AI",
                    "content": response["response"],
                    "metrics": {
                        "latency": response["latency"],
                        "euro_cost": response["euro_cost"],
                        "energy_usage": response["energy_usage"],
                        "gwp": response["gwp"],
                    },
                }
            )
        else:
            # Définition du message de sécurité
            message = (
                "Votre message n'a pas été traité pour des raisons de sécurité. "
                "Veuillez reformuler votre message."
            )

            # Affichage du message de sécurité
            with self.chat_container.chat_message("Guardian", avatar="⛔"):
                st.write_stream(stream_text(message))

            # Ajout du message de sécurité à l'historique de la conversation
            st.session_state["chats"][self.selected_chat].append(
                {
                    "role": "Guardian",
                    "content": message,
                }
            )

        # Si c'est le premier message envoyé, alors génération du nom de la conversation
        if len(st.session_state["chats"][self.selected_chat]) == 2:
            print(self.selected_chat)
            self.generate_chat_name(message)

    @st.dialog("Ajouter des fichiers PDF")
    def upload_files_dialog(self):
        """
        Ouvre une boîte de dialogue pour ajouter des fichiers PDF.
        """

        # Espace pour ajouter des fichiers
        uploaded_files = st.file_uploader(
            "NULL",
            label_visibility="collapsed",
            type=["pdf"],
            accept_multiple_files=True,
        )

        # Bouton pour ajouter et traiter les fichiers sélectionnés
        if st.button(
            "Ajouter les fichiers sélectionnés",
            icon=":material/upload_file:",
            disabled=not uploaded_files,
        ):
            with st.status(
                "**Ajout de(s) fichier(s) en cours... Ne fermez pas la fenêtre !**",
                expanded=True,
            ) as status:
                # Lecture du contenu de chaque fichier PDF
                documents = {}
                for file in uploaded_files:
                    st.write(f"Ajout du fichier {file.name}...")
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    documents[file.name] = text
                status.update(
                    label="**Les fichiers ont été ajoutés avec succès ! "
                    "Vous pouvez maintenant fermer la fenêtre.**",
                    state="complete",
                    expanded=False,
                )
