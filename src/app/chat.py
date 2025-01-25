import streamlit as st
import PyPDF2

from src.app.components import stream_text

class Chat:
    def __init__(self, selected_chat : str, initial_question : str = None):
        """
        Initialise la classe Chat avec le chat sélectionné et la question initiale si disponible.

        Args:
            selected_chat (str): Chat sélectionné pour la conversation.
            initial_question (str, optionnel): Question initiale à poser à l'IA. None par défaut.
        """
        
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

        # Stockage de la question initiale
        self.initial_question = initial_question
        st.session_state['initial_question'] = None

    def run(self):
        """
        Lance l'affichage du chat avec l'IA Mistral.
        """
        
        # Avertissement si la clé API Mistral n'est pas présente
        if st.session_state['found_mistral_api'] == False:
            # Affichage d'un message d'erreur
            st.error("**Conversation avec l'IA indisponible :** Votre clé d'API Mistral est introuvable.", icon=":material/error:")

        # Mise en page du chat avec l'IA
        header_container = st.container()
        self.chat_container = header_container.container(height=500)

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
        # Si une question initiale est présente, l'envoyer automatiquement
        if self.initial_question and not st.session_state["chats"][self.selected_chat]:
            self.handle_user_message(self.initial_question)

        # Mise en page de l'interraction avec l'IA
        cols = header_container.columns([3, 10, 1])

        # Choix du modèle [TEMP]
        with cols[0]:
            st.selectbox("NULL", label_visibility="collapsed", options=["Option 1", "Option 2", "Option 3"], index=0, disabled=not st.session_state.get('found_mistral_api', False))

        # Zone de saisie pour le chat avec l'IA [TEMP]
        with cols[1]:
            if message := st.chat_input(placeholder="Écrivez votre message", key=f"chat_input_{self.selected_chat}", disabled=not st.session_state.get('found_mistral_api', False)):
                if message.strip():
                    self.handle_user_message(message)

        # Bouton pour ajouter un fichier
        with cols[2]:
            if st.button("", icon=":material/attach_file:", disabled=not st.session_state.get('found_mistral_api', False)):
                self.upload_files_dialog()

    def handle_user_message(self, message: str):
        """
        Gère le message de l'utilisateur et envoie une requête à l'IA Mistral.

        Args:
            message (str): Message de l'utilisateur.
        """
        
        # Affichage du nouveau message de l'utilisateur
        with self.chat_container.chat_message("User", avatar="👤"):
            st.write(message)

        # Ajout du message à l'historique de la conversation
        st.session_state["chats"][self.selected_chat].append({"role": "User", "content": message})

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
        #     history=st.session_state["chats"][self.selected_chat]
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
        with self.chat_container.chat_message("AI", avatar="✨"):
            st.write_stream(stream_text(response["response"]))
            st.markdown(
                f"📶 *Latence : {response['latency']:.2f} secondes* | "
                f"💲 *Coût : {response['euro_cost']:.6f} €* | "
                f"⚡ *Utilisation énergétique : {response['energy_usage']} kWh* | "
                f"🌡️ *Potentiel de réchauffement global : {response['gwp']} kgCO2eq*"
            )

        # Ajout de la réponse de l'IA à l'historique de la conversation
        st.session_state["chats"][self.selected_chat].append({
            "role": "AI",
            "content": response["response"],
            "metrics": {
                "latency": response['latency'],
                "euro_cost": response['euro_cost'],
                "energy_usage": response['energy_usage'],
                "gwp": response['gwp']
            }
        })

    @st.dialog("Ajouter des fichiers PDF")
    def upload_files_dialog(self):
        """
        Ouvre une boîte de dialogue pour ajouter des fichiers PDF.
        """

        # Affichage de l'espace pour ajouter des fichiers
        uploaded_files = st.file_uploader("NULL", label_visibility="collapsed", type=['pdf'], accept_multiple_files=True)
        
        # Bouton pour ajouter et traiter les fichiers sélectionnés
        if st.button("Ajouter les fichiers sélectionnés", icon=":material/upload_file:", disabled=not uploaded_files):
            with st.status("**Ajout de(s) fichier(s) en cours... Ne fermez pas la fenêtre !**", expanded=True) as status:
                # Lecture du contenu de chaque fichier PDF
                documents = {}
                for file in uploaded_files:
                    st.write(f"Ajout du fichier {file.name}...")
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    documents[file.name] = text
                status.update(label="**Les fichiers ont été ajoutés avec succès ! Vous pouvez maintenant fermer la fenêtre.**", state="complete", expanded=False)