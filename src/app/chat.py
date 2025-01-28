"""
Ce fichier définit la classe Chat pour gérer les interractions avec l'IA.
"""

import os
import streamlit as st
import PyPDF2
import wikipedia

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

        # Définition du prompt
        prompt = (
            "Tu es une intelligence artificielle spécialisée dans l'aide aux élèves à l'école. "
            "Génère 5 questions courtes dans différentes matières sans les préciser, "
            "qu'un élève pourrait te poser sur une notion de cours. "
            "Répond uniquement en donnant les 5 questions sous forme de liste de tirets, "
            "sans explication supplémentaire."
        )

        # Génération des questions
        response = self.llm.generate(
            prompt=prompt,
            provider="mistral",
            model="mistral-large-latest",
            temperature=0.7,
            max_tokens=1000
        )

        # Récupération des questions
        questions = response["response"].split('\n')
        return [q.strip("- ").strip() for q in questions[:5]]

    def generate_chat_name(self, initial_message: str):
        """
        Génère un nom pour la conversation en fonction du premier message avec l'IA.

        Args:
            initial_message (str): Message initial de la conversation.
        """

        # 5 tentatives pour générer un nom de conversation
        for _ in range(5):
            # Définition du prompt
            prompt = (
                "Tu es une intelligence artificielle spécialisée "
                "dans la création de nom de conversation. "
                "En te basant sur le texte suivant, qui est le premier message de la conversation, "
                "propose un nom d'un maximum de 30 caractères pour cette conversation. "
                "Répond uniquement en donnant le nom de la conversation "
                "sans explication supplémentaire. "
                f"Voici le texte : {initial_message}"
            )

            # Génération du nom de la conversation
            response = self.llm.generate(
                prompt=prompt,
                provider="mistral",
                model="mistral-large-latest",
                temperature=0.7,
                max_tokens=100
            )

            # Récupération du nom de la conversation
            generated_name = response["response"].strip()

            # Vérification de la conformité du nom de la conversation
            if len(generated_name) > 30:
                continue
            if generated_name in st.session_state["chats"]:
                continue

            # Changement du nom de la conversation
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

        # Initialisation de la variable de modification des paramètres du modèle
        if "modified_model_params" not in st.session_state:
            st.session_state["modified_model_params"] = False

        # Initialisation de l'état de la recherche internet
        if "internet_search_active" not in st.session_state:
            st.session_state["internet_search_active"] = False

        # Affichage de l'historique de la conversation
        for idx, message in enumerate(st.session_state["chats"][self.selected_chat]):
            # Affichage des messages de l'utilisateur
            if message["role"] == "User":
                with self.chat_container.chat_message(message["role"], avatar="👤"):
                    st.write(message["content"])

            # Affichage des messages de l'IA
            elif message["role"] == "AI":
                with self.chat_container.chat_message(message["role"], avatar="✨"):
                    st.write(message["content"])
                    metrics = message["metrics"]
                    st.pills(
                        label="NULL",
                        options=[
                            f"📶 {metrics['latency']:.2f} secondes",
                            f"💲 {metrics['euro_cost']:.6f} €",
                            f"⚡ {metrics['energy_usage']} kWh",
                            f"🌡️ {metrics['gwp']} kgCO2eq",
                        ],
                        label_visibility="collapsed",
                        key=idx
                    )

            # Affichage des messages de sécurité
            elif message["role"] == "Guardian":
                with self.chat_container.chat_message(message["role"], avatar="🛡️"):
                    st.write(message["content"])

        # Si une question initiale est présente, l'envoyer automatiquement
        if self.initial_question and not st.session_state["chats"][self.selected_chat]:
            self.handle_user_message(self.initial_question)

        # Mise en page de l'interraction avec l'IA
        cols = self.header_container.columns([1, 13, 1, 1])

        # Paramètres du modèle
        with cols[0]:
            if st.button(
                "",
                icon=":material/tune:",
                disabled=not st.session_state.get("found_api_keys", False),
            ):
                self.settings_dialog()
            if st.session_state["modified_model_params"] is True:
                st.toast(
                    "Paramètres de l'IA modifiés avec succès !",
                    icon=":material/check_circle:"
                )
                st.session_state["modified_model_params"] = False

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
                disabled=(
                    not st.session_state.get("found_api_keys", False) or
                    st.session_state.get("internet_search_active", False)
                ),
            ):
                self.upload_files_dialog()

        # Mode recherche internet
        with cols[3]:
            if st.button(
                "",
                icon=":material/language:",
                disabled=not st.session_state.get("found_api_keys", False),
                type="primary" if st.session_state["internet_search_active"] else "secondary"
            ):
                if st.session_state["internet_search_active"] is True:
                    st.session_state["internet_search_active"] = False
                else:
                    st.session_state["internet_search_active"] = True
                st.rerun()

        # Message d'avertissement
        st.write(
            ":grey[*SISE Classmate peut faire des erreurs. "
            "Envisagez de vérifier les informations importantes "
            "et n'envoyez pas d'informations confidentielles.*]"
        )

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
        security_message = (
            "Votre message a été bloqué car il ne respecte pas nos conditions d'utilisation."
        )

        # Validation du message de l'utilisateur
        is_valid_message = security_manager.validate_input()

        # Si le message de utilisateur est autorisé
        if is_valid_message is True:
            # Si le mode recherche internet est activé
            if st.session_state["internet_search_active"] is True:
                # Recherche sur Wikipedia
                wiki_summary = self.fetch_wikipedia_data(message)

                # Enrichissement du message
                message = (
                    "Tu es une intelligence artificielle spécialisée dans "
                    "l'aide aux élèves à l'école, si le message ne concerne pas "
                    "une question de cours, alors tu renvoi seulement et uniquement le mot 'Guardian'. "
                    f"Voici le message de l'utilisateur : {message}."
                    "Pour répondre au message suivant, nous te fournissons du contenu "
                    "provenant d'un recherche sur Wikipedia "
                    f"afin de te donner des informations sur le sujet : {wiki_summary}."
                )
            # Si le mode recherche internet n'est pas activé
            else:
                # Enrichissement du message
                message = (
                    "Tu es une intelligence artificielle spécialisée dans "
                    "l'aide aux élèves à l'école, si le message ne concerne pas "
                    "une question de cours, alors tu renvoi seulement et uniquement le mot 'Guardian'. "
                    f"Voici le message de l'utilisateur : {message}."
                )

            # Récupération des paramètres du modèle
            model_params = self.llm.get_model_config()

            # Envoi du message et récupération de la réponse de l'IA
            response = self.llm.generate(
                prompt=message,
                provider=model_params["current_provider"],
                model=model_params["current_model"],
                temperature=model_params["current_temperature"],
                max_tokens=10000
            )

            # Si l'IA a renvoyé le mot "Guardian"
            if response["response"].strip() == "Guardian":
                # Affichage du message de sécurité
                with self.chat_container.chat_message("Guardian", avatar="🛡️"):
                    st.write_stream(stream_text(security_message))

                # Ajout du message de sécurité à l'historique de la conversation
                st.session_state["chats"][self.selected_chat].append(
                    {
                        "role": "Guardian",
                        "content": security_message,
                    }
                )
                return

            # Affichage de la réponse de l'IA
            with self.chat_container.chat_message("AI", avatar="✨"):
                st.write_stream(stream_text(response["response"]))
                st.pills(
                    label="NULL",
                    options=[
                        f"📶 {response['latency']:.2f} secondes",
                        f"💲 {response['euro_cost']:.6f} €",
                        f"⚡ {response['energy_usage']} kWh",
                        f"🌡️ {response['gwp']} kgCO2eq",
                    ],
                    label_visibility="collapsed"
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
                    "internet_search": st.session_state["internet_search_active"],
                    "model_used": model_params["current_model"]
                }
            )
        else:
            # Affichage du message de sécurité
            with self.chat_container.chat_message("Guardian", avatar="🛡️"):
                st.write_stream(stream_text(security_message))

            # Ajout du message de sécurité à l'historique de la conversation
            st.session_state["chats"][self.selected_chat].append(
                {
                    "role": "Guardian",
                    "content": security_message,
                }
            )

        # Si c'est le premier message envoyé, alors génération du nom de la conversation
        if len(st.session_state["chats"][self.selected_chat]) == 2:
            print(self.selected_chat)
            self.generate_chat_name(st.session_state["chats"][self.selected_chat][0]["content"])

    @st.dialog("Paramètres de l'IA")
    def settings_dialog(self):
        """
        Ouvre une boîte de dialogue pour configurer les paramètres de l'IA.
        """

        # Récupération de la configuration actuelle
        config = self.llm.get_model_config()
        providers = config["providers"]
        provider_options = list(providers.keys())

        # Paramètrage du fournisseur
        selected_provider = st.selectbox(
            label="Fournisseur",
            options=provider_options,
            index=provider_options.index(config["current_provider"]),
            help=(
                "Chaque fournisseur propose des modèles avec des optimisations spécifiques, "
                "des fonctionnalités uniques, ou des performances adaptées à certains cas d'usage."
            )
        )

        # Personnalisation de l'aide en fonction du fournisseur
        if selected_provider == "mistral":
            models_help = (
                "- :material/energy_savings_leaf: **ministral-8b-latest :** "
                "modèle généraliste de taille moyenne, "
                "équilibré pour des tâches variées avec des performances optimisées.\n"
                "- :material/energy_savings_leaf::material/energy_savings_leaf: "
                "**mistral-large-latest :** modèle de grande capacité, idéal pour des cas "
                "d'usage nécessitant des réponses complexes et détaillées.\n"
                "- :material/energy_savings_leaf::material/energy_savings_leaf:"
                ":material/energy_savings_leaf: **codestral-latest :** "
                "modèle spécialisé pour la génération de code "
                "et les tâches techniques, parfait pour les développeurs."
            )
        elif selected_provider == "gemini":
            models_help = (
                "- :material/energy_savings_leaf: **gemini-1.5-flash-8b :** "
                "modèle rapide et compact, conçu pour des "
                "interactions rapides sans sacrifier la qualité des réponses.\n"
                "- :material/energy_savings_leaf::material/energy_savings_leaf: "
                "**gemini-1.5-flash :** modèle optimisé pour la vitesse, "
                "offrant un bon compromis entre réactivité et précision.\n"
                "- :material/energy_savings_leaf::material/energy_savings_leaf:"
                ":material/energy_savings_leaf: **gemini-1.5-pro :** "
                "modèle avancé avec des capacités professionnelles, "
                "idéal pour les analyses approfondies et des applications exigeantes."
            )
        else:
            models_help = "Aucune information sur les modèles est disponible."

        # Explication de l'indicateur d'impact énergétique et écologique
        models_help = models_help + (
            "\n\n :material/energy_savings_leaf: *indique l'impact énergétique "
            "et écologique du modèle : moins il y a de symboles, plus le modèle "
            "est respectueux de l'environnement et économe en énergie.*"
        )

        # Paramètrage du modèle
        models = providers[selected_provider]["models"]
        selected_model = st.selectbox(
            label="Modèle",
            options=models,
            index=(
                models.index(config["current_model"])
                if config["current_model"] in models else 0
            ),
            help=models_help
        )

        # Paramètrage de la température
        selected_temperature = st.slider(
            "Température (%)",
            min_value=0.0,
            max_value=100.0,
            value=config["current_temperature"] * 100,
            step=1.0,
            help=(
                "Contrôle la variabilité des réponses générées par le modèle. "
                "Une **température basse** (proche de 0) rend les réponses plus "
                "**cohérentes et déterministes**, tandis qu'une **température élevée** "
                "(proche de 100) favorise des réponses plus **créatives et variées**."
            )
        )
        selected_temperature /= 100.0

        # Enregistrement des paramètres
        if st.button("Enregistrer", icon=":material/save:"):
            self.llm.switch_provider(selected_provider, selected_model, selected_temperature)
            st.session_state["modified_model_params"] = True
            st.rerun()

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

    def fetch_wikipedia_data(self, query: str) -> str:
        """
        Recherche des informations sur Wikipedia pour la requête donnée.

        Args:
            query (str): La requête de recherche.

        Returns:
            str: Résumé des informations trouvées.
        """
        try:
            # Récupération des informations sur Wikipedia en français
            wikipedia.set_lang("fr")
            summary = wikipedia.summary(query)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Le message est ambigu, voici les suggestions de Wikipedia : {e.options[:5]}"
        except wikipedia.exceptions.PageError:
            return "Wikipedia n'a pas trouvé d'informations correspondant au message"
