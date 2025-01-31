"""
Ce fichier définit la classe Chat pour gérer les interractions avec l'IA.
"""

import streamlit as st
import PyPDF2

from src.app.components import stream_text, convert_to_json
from src.pipelines import EnhancedLLMSecurityManager, PDFPipeline
from src.llm.rag import RAG

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
        # Initialisation du fournisseur d'IA
        if "AI_provider" not in st.session_state:
            st.session_state["AI_provider"] = "mistral"

        # Initialisation du modèle d'IA
        if "AI_model" not in st.session_state:
            st.session_state["AI_model"] = "ministral-8b-latest"

        # Initialisation de la température du modèle d'IA
        if "AI_temperature" not in st.session_state:
            st.session_state["AI_temperature"] = 0.7

        # Récupération du chat sélectionné
        self.selected_chat = selected_chat

        # Si ce ne sont pas les suggestions
        if selected_chat != "suggestions":
            # Stockage de la question initiale
            self.initial_question = initial_question
            st.session_state["initial_question"] = None

            # Mise en page du chat avec l'IA
            self.header_container = st.container()
            self.chat_container = self.header_container.container(height=500)

        # Initialisation du Guardian
        if "GUARDIAN" not in st.session_state:
            st.session_state["GUARDIAN"] = EnhancedLLMSecurityManager()

        # Initialisation du pipeline de traitement des fichiers PDF
        if "PDF_PIPELINE" not in st.session_state:
            st.session_state["PDF_PIPELINE"] = PDFPipeline()

        # Si les clés d'API sont trouvées
        if st.session_state["found_api_keys"] is True:
            # Initialisation du LLM
            if "LLM" not in st.session_state:
                st.session_state["LLM"] = RAG(
                    max_tokens=7000,
                    top_n=3
                )
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

        # Génération des questions
        response = st.session_state["LLM"](
            provider="mistral",
            model="mistral-large-latest",
            temperature=0.7,
            prompt_type="suggestions"
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
            # Génération du nom de la conversation
            response = st.session_state["LLM"](
                provider="mistral",
                model="mistral-large-latest",
                temperature=0.7,
                prompt_type="chat_name",
                message=initial_message
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
        for idx, message in enumerate(st.session_state["chats"][self.selected_chat]["messages"]):
            # Affichage des messages de l'utilisateur
            if message["role"] == "User":
                with self.chat_container.chat_message(message["role"], avatar="👤"):
                    st.markdown(message["content"])

            # Affichage des messages de l'IA
            elif message["role"] == "AI":
                with self.chat_container.chat_message(message["role"], avatar="✨"):
                    st.markdown(message["content"])
                    metrics = message["metrics"]
                    st.pills(
                        label="NULL",
                        options=[
                            f"📶 {metrics['latency']:.2f} secondes",
                            f"💲 {metrics['euro_cost']:.7f} €",
                            f"⚡ {metrics['energy_usage']:.7f} kWh",
                            f"🌡️ {metrics['gwp']:.7f} kgCO2eq",
                        ],
                        label_visibility="collapsed",
                        key=idx
                    )

            # Affichage des messages de sécurité
            elif message["role"] == "Guardian":
                with self.chat_container.chat_message(message["role"], avatar="🛡️"):
                    st.write(message["content"])

        # Si une question initiale est présente, l'envoyer automatiquement
        if self.initial_question and not st.session_state["chats"][self.selected_chat]["messages"]:
            self.handle_user_message(self.initial_question)

        # Si une demande d'explication de réponse de quiz est présente, l'envoyer automatiquement
        if 'quiz_answer_explanation' in st.session_state:
            message = st.session_state.pop('quiz_answer_explanation')
            self.handle_user_message(message)

        # Mise en page de l'interraction avec l'IA
        cols = self.header_container.columns([1, 12, 1, 1, 1])

        # Paramètres du modèle
        with cols[0]:
            if st.button(
                "",
                icon=":material/tune:",
                disabled=not st.session_state.get("found_api_keys", False),
                use_container_width=True
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

        # Bouton pour ajouter un fichier PDF
        with cols[2]:
            if st.button(
                "",
                icon=":material/attach_file:",
                disabled=(
                    not st.session_state.get("found_api_keys", False) or
                    st.session_state.get("internet_search_active", False)
                ),
                use_container_width=True
            ):
                self.upload_files_dialog()

        # Mode recherche internet
        with cols[3]:
            if st.button(
                "",
                icon=":material/language:",
                disabled=not st.session_state.get("found_api_keys", False),
                use_container_width=True,
                type="primary" if st.session_state["internet_search_active"] else "secondary"
            ):
                if st.session_state["internet_search_active"] is True:
                    st.session_state["internet_search_active"] = False
                else:
                    st.session_state["internet_search_active"] = True
                st.rerun()

        # Mode quizz
        with cols[4]:
            if st.button(
                "",
                icon=":material/check_box:",
                disabled=(
                    not st.session_state.get("found_api_keys", False) or
                    st.session_state["chats"][self.selected_chat]["messages"] == []
                ),
                use_container_width=True
            ):
                self.generate_quiz()

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
        st.session_state["chats"][self.selected_chat]["messages"].append(
            {"role": "User", "content": message}
        )

        # Définition du message de sécurité
        security_message = (
            "Votre message a été bloqué car il ne respecte pas nos conditions d'utilisation. "
            "Si vous estimez qu'il a été bloqué par erreur, veuillez essayer de le reformuler."
        )

        # Validation du message de l'utilisateur
        is_valid_message = st.session_state["GUARDIAN"].validate_input(message)

        # Si le message de utilisateur est autorisé
        if is_valid_message is True:
            # Définition du type de génération
            if st.session_state["internet_search_active"] is True:
                prompt_type = "internet_chat"
            else:
                prompt_type = "chat"

            # Préparation de l'affichage du message de l'IA
            with self.chat_container.chat_message("AI", avatar="✨"):
                with st.spinner("Je réfléchis..."):
                # Envoi du message et récupération de la réponse de l'IA
                    response = st.session_state["LLM"](
                        provider=st.session_state["AI_provider"],
                        model=st.session_state["AI_model"],
                        temperature=st.session_state["AI_temperature"],
                        prompt_type=prompt_type,
                        message=message,
                        message_history=st.session_state["chats"][self.selected_chat]["messages"],
                        ressources=st.session_state["chats"][self.selected_chat]["document_ids"]
                    )

                # Si l'IA a renvoyé le mot "Guardian"
                if response["response"].strip() == "Guardian":
                    # Affichage du message de sécurité
                    with self.chat_container.chat_message("Guardian", avatar="🛡️"):
                        st.write_stream(stream_text(security_message))

                    # Ajout du message de sécurité à l'historique de la conversation
                    st.session_state["chats"][self.selected_chat]["messages"].append(
                        {
                            "role": "Guardian",
                            "content": security_message,
                        }
                    )
                    st.rerun()

                # Affichage de la réponse de l'IA
                st.write_stream(stream_text(response["response"]))
                st.pills(
                    label="NULL",
                    options=[
                        f"📶 {response['latency']:.2f} secondes",
                        f"💲 {response['euro_cost']:.7f} €",
                        f"⚡ {response['energy_usage']:.7f} kWh",
                        f"🌡️ {response['gwp']:.7f} kgCO2eq",
                    ],
                    label_visibility="collapsed"
                )

            # Ajout de la réponse de l'IA à l'historique de la conversation
            st.session_state["chats"][self.selected_chat]["messages"].append(
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
                    "model_used": st.session_state["AI_model"]
                }
            )
        else:
            # Affichage du message de sécurité
            with self.chat_container.chat_message("Guardian", avatar="🛡️"):
                st.write_stream(stream_text(security_message))

            # Ajout du message de sécurité à l'historique de la conversation
            st.session_state["chats"][self.selected_chat]["messages"].append(
                {
                    "role": "Guardian",
                    "content": security_message,
                }
            )

        # Si c'est le premier message envoyé, alors génération du nom de la conversation
        if len(st.session_state["chats"][self.selected_chat]["messages"]) == 2:
            self.generate_chat_name(st.session_state["chats"][self.selected_chat]["messages"][0]["content"])

    @st.dialog("Paramètres de l'IA")
    def settings_dialog(self):
        """
        Ouvre une boîte de dialogue pour configurer les paramètres de l'IA.
        """

        # Récupération de la configuration actuelle
        current_provider = st.session_state["AI_provider"]
        current_model = st.session_state["AI_model"]
        current_temperature = st.session_state["AI_temperature"]

        # Paramètrage du fournisseur
        selected_provider = st.selectbox(
            label="Fournisseur",
            options=["mistral", "gemini"],
            index=["mistral", "gemini"].index(current_provider),
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
        else:
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

        # Explication de l'indicateur d'impact énergétique et écologique
        models_help = models_help + (
            "\n\n :material/energy_savings_leaf: *indique l'impact énergétique "
            "et écologique du modèle : moins il y a de symboles, plus le modèle "
            "est respectueux de l'environnement et économe en énergie.*"
        )

        # Paramètrage du modèle
        if selected_provider == "mistral":
            models = ["ministral-8b-latest", "mistral-large-latest", "codestral-latest"]
        else:
            models = ["gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-1.5-pro"]

        # Assurer que current_model est dans models
        if current_model in models:
            default_index = models.index(current_model)
        else:
            default_index = 0

        selected_model = st.selectbox(
            label="Modèle",
            options=models,
            index=default_index,
            help=models_help
        )

        # Paramètrage de la température
        selected_temperature = st.slider(
            "Température (%)",
            min_value=0.0,
            max_value=100.0,
            value=current_temperature * 100,
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
        if st.button("Enregistrer", icon=":material/save:", use_container_width=True):
            st.session_state["AI_provider"] = selected_provider
            st.session_state["AI_model"] = selected_model
            st.session_state["AI_temperature"] = selected_temperature
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
            use_container_width=True
        ):
            with st.status(
                "**Ajout des fichiers en cours... Ne fermez pas la fenêtre !**",
                expanded=True,
            ) as status:
                # Lecture du contenu de chaque fichier PDF
                for file in uploaded_files:
                    st.write(f"Traitement du fichier {file.name} en cours...")
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    document_ids = st.session_state["PDF_PIPELINE"].process_txt(text)
                    for document_id in document_ids:
                        st.session_state["chats"][self.selected_chat]["document_ids"].append(document_id)
                    st.write(f"Traitement du fichier {file.name} terminé !")
                status.update(
                    label="**Les fichiers ont été ajoutés avec succès ! "
                    "Vous pouvez maintenant fermer la fenêtre.**",
                    state="complete",
                    expanded=False,
                )

    @st.dialog("Quiz", width="large")
    def generate_quiz(self):
        """
        Génère un quiz avec des questions sur le sujet donné, 
        sans recharger l'application à chaque interaction.
        """

        # Paramétrage du quiz
        with st.container(border=True):
            nb_questions = st.slider(
                "Nombre de questions",
                min_value=1,
                max_value=10,
                value=5,
                step=1
            )
            if st.button("Créer un quiz", use_container_width=True):
                # Réinitialisation des données du quiz
                st.session_state['quiz_data'] = None
                st.session_state['quiz_answers'] = {}
                st.session_state['quiz_submitted'] = False
                st.session_state['quiz_score'] = 0
                st.session_state['quiz_total'] = 0
                st.session_state['quiz_results'] = []

                # Génération des questions du quiz
                with st.spinner("Création du quiz..."):
                    response = st.session_state["LLM"](
                        provider="mistral",
                        model="mistral-large-latest",
                        temperature=0.7,
                        prompt_type="quizz",
                        message_history=st.session_state["chats"].get(self.selected_chat, {}).get("messages", []),
                        nb_questions=nb_questions
                    )

                    # Conversion des données du quiz
                    try :
                        st.session_state['quiz_data'] = convert_to_json(response["response"])
                        st.session_state['quiz_answers'] = {}
                        st.session_state['quiz_submitted'] = False
                    except Exception:
                        st.error(
                            "Une erreur est survenue lors de la création du quiz. "
                            "Veuillez réessayer."
                        )

        if 'quiz_data' in st.session_state and st.session_state['quiz_data'] is not None:
            quiz_col, result_col = st.columns([3, 2])

            with quiz_col:
                with st.form(key="quiz_form"):
                    # Affichage des questions du quiz
                    for idx, question_data in enumerate(st.session_state['quiz_data']):
                        st.subheader(f"Question {idx + 1}")
                        st.write(question_data["question"])

                        options = question_data["options"]
                        st.session_state['quiz_answers'][idx] = st.radio(
                            "Choisissez une réponse :",
                            options=options,
                            index=0,
                            key=f"question_{idx}"
                        )

                    # Bouton de validation des réponses
                    if st.form_submit_button(
                        "Valider les réponses",
                        disabled=st.session_state['quiz_submitted'],
                        use_container_width=True
                    ):
                        score, total, results = self.evaluate_quiz(
                            st.session_state['quiz_data'],
                            st.session_state['quiz_answers']
                        )
                        st.session_state['quiz_score'] = score
                        st.session_state['quiz_total'] = total
                        st.session_state['quiz_results'] = results
                        st.session_state['quiz_submitted'] = True
                        st.rerun(scope="fragment")

            # Affichage des résultats du quiz
            with result_col:
                with st.container(border=True):
                    st.subheader("Résultats")
                    if st.session_state.get('quiz_submitted'):
                        for idx, res in enumerate(st.session_state['quiz_results']):
                            if res["correct"]:
                                st.success(
                                    f"✅ {res['question']}\n\n"
                                    f"**Bonne réponse : {res['user_answer']}**"
                                )
                            else:
                                st.error(
                                    f"❌ {res['question']}\n\n"
                                    f"Votre réponse : {res['user_answer']}\n\n"
                                    f"**Bonne réponse : {res['correct_answer']}**"
                                )
                            if st.button(
                                "Expliquer la réponse",
                                key=f"explain_{idx}",
                                use_container_width=True
                            ):
                                st.session_state['quiz_answer_explanation'] = (
                                    f"Pourquoi la bonne réponse à la question **{res['question']}**\n"
                                    f"est : **{res['correct_answer']}**\n"
                                    
                                )
                                st.session_state['close_quiz_dialog'] = True
                                st.rerun()
                        st.info(
                            f"🎯 **Score final : {st.session_state['quiz_score']} "
                            f"/ {st.session_state['quiz_total']}**"
                        )

                        if st.session_state['quiz_score'] == st.session_state['quiz_total']:
                            st.balloons()
                    else:
                        st.info(
                            "Veuillez valider vos réponses pour afficher les résultats.",
                            icon=":material/info:"
                        )

   
    def evaluate_quiz(self, quiz_data : list, user_answers : dict) -> tuple:
        """
        Évalue les réponses du quiz et retourne le score final.

        Args:
            quiz_data (list): Liste des questions avec les bonnes réponses.
            user_answers (dict): Réponses de l'utilisateur.

        Returns:
            tuple: (score, nombre total de questions, liste des résultats détaillés)
        """

        # Initialisation des variables
        score = 0
        total = len(quiz_data)
        results = []

        # Évaluation des réponses
        for idx, question_data in enumerate(quiz_data):
            correct_answer = question_data["answer"]
            user_answer = user_answers.get(idx, None)

            is_correct = user_answer == correct_answer
            if is_correct:
                score += 1

            # Ajout des résultats
            results.append({
                "question": question_data["question"],
                "user_answer": user_answer if user_answer else "Aucune réponse",
                "correct_answer": correct_answer,
                "correct": is_correct
            })

        return score, total, results
