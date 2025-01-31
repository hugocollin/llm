"""
Ce fichier contient les fonctions nécessaires pour l'affichage de l'interface de l'application.
"""

import os
import time
import json
import streamlit as st
import plotly.express as px
from dotenv import find_dotenv, load_dotenv

def load_api_keys():
    """
    Fonction pour charger et stocker les clés API.
    """
    # Recherche des clés API si l'application est utilisé en local
    try:
        load_dotenv(find_dotenv())
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    # Sinon recherche des clés API si l'application est utilisé en ligne
    except FileNotFoundError:
        mistral_api_key = st.secrets["MISTRAL_API_KEY"]
        gemini_api_key = st.secrets["GEMINI_API_KEY"]

    # Stockage du statut de recherche des clés API
    if mistral_api_key and gemini_api_key:
        st.session_state["found_api_keys"] = True
    else:
        st.session_state["found_api_keys"] = False

def stream_text(text: str):
    """
    Fonction pour afficher le texte progressivement.

    Args:
        text (str): Texte à afficher progressivement.
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

def convert_to_json(response : str) -> dict:
    """
    Fonction pour convertir une réponse en JSON.

    Args:
        response (str): Réponse à convertir en JSON.

    Returns:
        dict: Réponse convertie en JSON.
    """
    res = response.strip("```json\n").strip("\n```")
    return json.loads(res)

def create_new_chat():
    """
    Fonction pour créer une nouvelle conversation.
    """

    # Récupération des numéros de conversation existants
    existing_numbers = [
        int(name.split(" ")[1])
        for name in st.session_state["chats"].keys()
        if name.startswith("Conversation ") and name.split(" ")[1].isdigit()
    ]

    # Recherche du prochain numéro de conversation disponible
    n = 1
    while n in existing_numbers:
        n += 1

    # Création de la nouvelle conversation
    new_chat_name = f"Conversation {n}"
    st.session_state["chats"][new_chat_name] = {
        "messages": [],
        "document_ids": []
    }
    st.session_state["selected_chat"] = new_chat_name

def select_chat(chat_name : str):
    """
    Fonction pour sélectionner une conversation.
    """
    st.session_state["selected_chat"] = chat_name

@st.dialog("Renommer la conversation")
def rename_chat(current_name : str):
    """
    Fonction pour renommer une conversation.

    Args:
        current_name (str): Nom actuel de la conversation.
    """

    # Saisie du nouveau nom de la conversation
    new_name = st.text_input(
        "Saisissez le nouveau nom de la conversation :",
        value=current_name,
        max_chars=30
    )

    if st.button("Enregistrer", icon=":material/save_as:", use_container_width=True):
        # Vérification de la conformité du nouveau nom
        if new_name in st.session_state["chats"] and new_name != current_name:
            st.error(
                "Ce nom de conversation existe déjà, veuillez en choisir un autre.",
                icon=":material/error:"
            )
        # Enregistrement du nouveau nom
        else:
            st.session_state["chats"][new_name] = st.session_state["chats"].pop(current_name)
            st.session_state["selected_chat"] = new_name
            if new_name != current_name:
                st.session_state["chat_renamed"] = True
            st.rerun()

def show_sidebar() -> str:
    """
    Fonction pour afficher la barre latérale de l'application.

    Returns:
        str: Nom de la conversation sélectionnée.
    """

    # Initialisation des conversations
    if "chats" not in st.session_state:
        st.session_state["chats"] = {}

    # Initialisation de la variable d'état de renommage de conversation
    if "chat_renamed" not in st.session_state:
        st.session_state["chat_renamed"] = False

    with st.sidebar:
        # Titre de l'application
        st.title("✨ SISE Classmate")

        cols = st.columns([1, 1, 3])

        # Bouton pour revenir à l'accueil
        with cols[0]:
            if st.button("", icon=":material/home:", use_container_width=True):
                st.session_state["selected_chat"] = None

        # Bouton pour afficher les informations sur l'application
        with cols[1]:
            if st.button("", icon=":material/info:", use_container_width=True):
                show_info_dialog()

        header_cols = st.columns([3, 1, 1])

        # Section des conversations
        with header_cols[0]:
            st.header("Conversations")

        # Bouton pour afficher les statistiques
        with header_cols[1]:
            st.write("")
            if st.button("", icon=":material/bar_chart:", use_container_width=True):
                show_stats_dialog()

        # Bouton pour ajouter un chat
        with header_cols[2]:
            st.write("")
            if st.button("", icon=":material/add_comment:", use_container_width=True):
                if len(st.session_state["chats"]) < 5:
                    create_new_chat()
                else:
                    st.toast(
                        "Nombre maximal de conversations atteint, "
                        "supprimez-en une pour en commencer une nouvelle",
                        icon=":material/feedback:",
                    )

        # Sélecteur d'espaces de discussion
        if st.session_state["chats"]:
            selected_chat = None
            for chat_name in list(st.session_state["chats"].keys()):
                btn_cols = st.columns([3, 1, 1])

                # Bouton pour sélectionner le chat
                with btn_cols[0]:
                    st.button(
                        f":material/forum: {chat_name}",
                        type="primary" if chat_name == st.session_state.get("selected_chat") else "secondary",
                        use_container_width=True,
                        on_click=select_chat,
                        args=(chat_name,)
                    )

                # Boutons pour renommer le chat
                with btn_cols[1]:
                    if st.button(
                        "",
                        icon=":material/edit:", key=f"rename_'{chat_name}'_button",
                        use_container_width=True
                    ):
                        rename_chat(chat_name)
                    if st.session_state["chat_renamed"] is True:
                        st.toast(
                            "Conversation renommée avec succès !",
                            icon=":material/check_circle:"
                        )
                        st.session_state["chat_renamed"] = False

                # Bouton pour supprimer le chat
                with btn_cols[2]:
                    if st.button(
                        "",
                        icon=":material/delete:", key=f"delete_'{chat_name}'_button",
                        use_container_width=True
                    ):
                        del st.session_state["chats"][chat_name]
                        del st.session_state[f"delete_'{chat_name}'_button"]
                        if st.session_state.get("selected_chat") == chat_name:
                            st.session_state["selected_chat"] = next(
                                iter(st.session_state["chats"]), None
                            )
                        st.rerun()
            return selected_chat
        else:
            # Message d'information si aucune conversation n'a été créée
            st.info(
                "Pour commencer une nouvelle conversation, "
                "cliquez sur le bouton :material/add_comment:",
                icon=":material/info:",
            )
            return None

@st.dialog("Informations sur l'application", width="large")
def show_info_dialog():
    """
    Fonction pour afficher les informations sur l'application.
    """

    # Information générale
    st.write(
        "**SISE Classmate est un assistant conversationnel spécialisé dans le "
        "domaine de l'éducation fonctionnant grâce aux modèles d'intelligence "
        "artificielle de Mistral et Gemini, que vous pouvez utiliser sur tous "
        "vos appareils, qu’il s’agisse de smartphones, tablettes ou ordinateurs.**"
    )
    st.write("Sur cette application vous pourrez :")

    with st.container(border=True):
        st.header("💬 Discuter avec l'IA")
        st.write(
            "Posez vos questions et obtenez des réponses précises et approfondies sur vos cours. "
            "L'IA, ayant accès à plus de 6 000 cours et à l'historique des messages précédemment "
            "envoyés dans la discussion, vous aide à réviser et à mieux comprendre les sujets "
            "abordés en classe."
        )

    cols = st.columns(2)
    with cols[0]:
        with st.container(border=True):
            st.header("💡 Obtenir des suggestions de messages")
            st.write(
                "L'IA génère automatiquement cinq suggestions de questions "
                "pour faciliter vos interactions et mieux formuler vos demandes."
            )
    with cols[1]:
        with st.container(border=True):
            st.header("⚙️ Paramétrer l'IA selon vos besoins")
            st.write(
                "Personnalisez les paramètres de l'IA, comme le fournisseur, "
                "le modèle et la température, pour une expérience adaptée à vos besoins."
            )

    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            st.header("📄 Ajouter des documents à la discussion")
            st.write(
                "Importez des fichiers PDF pour permettre à l'IA d'analyser "
                "leur contenu et d'enrichir ses réponses."
            )
    with cols[1]:
        with st.container(border=True):
            st.header("🌐 Enrichir les réponses grâce à internet")
            st.write(
                "Activez la recherche en ligne pour obtenir "
                "des réponses actualisées et plus pertinentes."
            )
    with cols[2]:
        with st.container(border=True):
            st.header("⛳ S'entraîner grâce à des quiz")
            st.write(
                "Générez des quiz interactifs basés sur le sujet de "
                "discussion pour tester et renforcer vos connaissances."
            )

    cols = st.columns(2)
    with cols[0]:
        with st.container(border=True):
            st.header("📊 Visualiser les statistiques de conversation")
            st.write(
                "Consultez des statistiques détaillées sur l'utilisation "
                "de l'application, incluant notamment la latence, le coût, "
                "l'énergie consommée et l'empreinte carbone des messages et plus encore."
            )
    with cols[1]:
        with st.container(border=True):
            st.header("✒️ Personnaliser les noms de conversation")
            st.write(
                "Obtenez automatiquement un nom pertinent pour chaque "
                "conversation ou personnalisez-le selon vos préférences."
            )

    with st.container(border=True):
        st.header("🛡️ Assurez un espace d'échange sécurisé grâce au Guardian")
        st.write(
            "Le Guardian est un module de sécurité qui bloque les messages inappropriés, "
            "non pertinents ou dangereux, en limitant les interactions aux sujets éducatifs, "
            "scolaires et de culture générale. Il protège également contre les attaques, "
            "telles que les tentatives d'injection SQL, garantissant un espace "
            "d'apprentissage sûr, sain et fiable."
        )

    # Crédits de l'application
    st.write(
        "*L'application est Open Source et disponible sur "
        "[GitHub](https://github.com/hugocollin/llm). "
        "Celle-ci a été développée par "
        "[KPAMEGAN Falonne](https://github.com/marinaKpamegan), "
        "[KARAMOKO Awa](https://github.com/karamoko17), "
        "[CISSE Lansana](https://github.com/lansanacisse) "
        "et [COLLIN Hugo](https://github.com/hugocollin), dans le cadre du Master 2 SISE.*"
    )

@st.dialog("Statistiques de conversation", width="large")
def show_stats_dialog():
    """
    Fonction pour afficher les statistiques globales ou détaillées par conversation.
    """
    # Récupération des noms des conversations
    conversations = list(st.session_state.get("chats", {}).keys())

    if conversations:
        # Sélection des conversations à analyser
        selected_conversations = st.pills(
            label="Sélectionnez les conversations à analyser :",
            options=conversations,
            selection_mode="multi",
            default=conversations
        )

        # Récupération des conversations à analyser
        if selected_conversations:
            chats_to_analyze = {
                k: v
                for k, v in st.session_state["chats"].items()
                if k in selected_conversations
            }

            # Initialisation des variables pour les statistiques
            total_user_messages = 0
            total_ai_messages = 0
            total_blocked_messages = 0
            total_latency = 0.0
            total_cost = 0.0
            total_energy = 0.0
            total_gwp = 0.0
            total_internet_search = 0
            model_counts = {}
            total_documents_imported = 0

            # Calcul des statistiques
            for chat in chats_to_analyze.values():
                for message in chat.get("messages", []):
                    if message["role"] == "User":
                        total_user_messages += 1
                    if message["role"] == "AI":
                        total_ai_messages += 1
                        metrics = message.get("metrics", {})
                        total_latency += metrics.get("latency", 0.0)
                        total_cost += metrics.get("euro_cost", 0.0)
                        total_energy += metrics.get("energy_usage", 0.0)
                        total_gwp += metrics.get("gwp", 0.0)
                        total_internet_search += 1 if message.get("internet_search") else 0
                        model = message.get("model_used", "Inconnu")
                        model_counts[model] = model_counts.get(model, 0) + 1
                    if message["role"] == "Guardian":
                        total_blocked_messages += 1
                document_ids = chat.get("document_ids", {})
                total_documents_imported += len(document_ids)

            average_latency = total_latency / total_ai_messages if total_ai_messages > 0 else 0

            # Option pour afficher les graphiques
            afficher_graphiques = st.toggle("Afficher les détails", False)

            # Affichage des graphiques
            if afficher_graphiques:
                conversation_names = list(chats_to_analyze.keys())
                
                # Graphique de la répartition du nombre total d'utilisations de chaque modèle d'IA
                with st.container(border=True):
                    st.header(
                        "**📊 Nombre total d'utilisations de chaque modèle d'IA**"
                    )

                    if total_ai_messages == 0:
                        st.info(
                            "Le graphique ne possède pas de données à afficher.",
                            icon=":material/info:"
                        )
                    else:
                        fig = px.pie(
                            names=list(model_counts.keys()),
                            values=list(model_counts.values()),
                            color_discrete_map=px.colors.qualitative.D3
                        )
                        fig.update_traces(
                            texttemplate='%{value} <br> %{percent}',
                            hovertemplate='<b>%{label} :</b> %{value} (%{percent})<extra></extra>'
                        )
                        st.plotly_chart(fig, key="models_chart")

                # Graphique de la répartition du nombre total de messages envoyés
                with st.container(border=True):
                    st.header(
                            f"**🗨️ Nombre total de messages envoyés : {total_user_messages}**"
                    )

                    if total_user_messages == 0:
                        st.info(
                            "Le graphique ne possède pas de données à afficher.",
                            icon=":material/info:"
                        )
                    else:
                        message_counts = [
                            sum(1 for message in chat.get("messages", []) if message.get("role") == "User")
                            for chat in chats_to_analyze.values()
                        ]
                        fig = px.pie(
                            names=conversation_names,
                            values=message_counts,
                            color_discrete_sequence=px.colors.sequential.Blues[::-1]
                        )
                        fig.update_traces(
                            texttemplate='%{value}<br>%{percent}',
                            hovertemplate=(
                                '<b>%{label} :</b> %{value} '
                                'messages (%{percent})<extra></extra>'
                            )
                        )
                        st.plotly_chart(fig, key="messages_chart")

                # Graphique de la répartition du nombre total de messages bloqués
                with st.container(border=True):
                    st.header(
                        "**🛡️ Nombre total de messages bloqués : "
                        f"{total_blocked_messages}**"
                    )

                    if total_blocked_messages == 0:
                        st.info(
                            "Le graphique ne possède pas de données à afficher.",
                            icon=":material/info:"
                        )
                    else:
                        blocked_message_counts = [0] * len(conversation_names)
                        for i, chat in enumerate(chats_to_analyze.values()):
                            for message in chat.get("messages", []):
                                if message["role"] == "Guardian":
                                    blocked_message_counts[i] += 1
                        fig = px.pie(
                            names=conversation_names,
                            values=blocked_message_counts,
                            color_discrete_sequence=px.colors.sequential.Purples[::-1]
                        )
                        fig.update_traces(
                            texttemplate='%{value}<br>%{percent}',
                            hovertemplate=(
                                '<b>%{label} :</b> %{value} '
                                'messages bloqués (%{percent})<extra></extra>'
                            )
                        )
                        st.plotly_chart(fig, key="blocked_messages_chart")

                # Graphique de la répartition du nombre total de documents importés
                with st.container(border=True):
                    st.header(
                        "**📄 Nombre total de documents importés : "
                        f"{total_documents_imported}**"
                    )

                    if total_documents_imported == 0:
                        st.info(
                            "Le graphique ne possède pas de données à afficher.",
                            icon=":material/info:"
                        )
                    else:
                        document_counts = [
                            len(chat.get("document_ids", []))
                            for chat in chats_to_analyze.values()
                        ]
                        fig = px.pie(
                            names=conversation_names,
                            values=document_counts,
                            color_discrete_sequence=px.colors.sequential.Oranges[::-1]
                        )
                        fig.update_traces(
                            texttemplate='%{value}<br>%{percent}',
                            hovertemplate=(
                                '<b>%{label} :</b> %{value} '
                                'documents importés (%{percent})<extra></extra>'
                            )
                        )
                        st.plotly_chart(fig, key="documents_chart")

                # Graphique de la répartition du nombre total d'utilisation du mode internet
                with st.container(border=True):
                    st.header(
                        "**🌐 Nombre total d'utilisations du mode internet : "
                        f"{total_internet_search}**"
                    )

                    if total_internet_search == 0:
                        st.info(
                            "Le graphique ne possède pas de données à afficher.",
                            icon=":material/info:"
                        )
                    else:
                        internet_search_counts = [0] * len(conversation_names)
                        for i, chat in enumerate(chats_to_analyze.values()):
                            for message in chat.get("messages", []):
                                if message.get("internet_search"):
                                    internet_search_counts[i] += 1
                        fig = px.pie(
                            names=conversation_names,
                            values=internet_search_counts,
                            color_discrete_sequence=px.colors.sequential.Mint[::-1]
                        )
                        fig.update_traces(
                            texttemplate='%{value}<br>%{percent}',
                            hovertemplate=(
                                '<b>%{label} :</b> %{value} '
                                'utilisations du mode internet (%{percent})<extra></extra>'
                            )
                        )
                        st.plotly_chart(fig, key="internet_search_chart")

                # Graphique de la latence moyenne
                with st.container(border=True):
                    st.header(
                        "**📶 Latence moyenne des réponses : "
                        f"{average_latency:.2f} secondes**"
                    )

                    if total_ai_messages == 0:
                        st.info(
                            "Le graphique ne possède pas de données à afficher.",
                            icon=":material/info:"
                        )
                    else:
                        latences = [
                            message["metrics"]["latency"]
                            for chat in chats_to_analyze.values()
                            for message in chat.get("messages", [])
                            if message["role"] == "AI" and "latency" in message.get("metrics", {})
                        ]
                        fig = px.histogram(
                            latences,
                            nbins=20,
                            labels={"value": "Latence (secondes)"},
                            color_discrete_sequence=px.colors.sequential.Bluered
                        )
                        fig.update_layout(
                            xaxis_title="Latence (secondes)",
                            yaxis_title="Nombre de réponses",
                            showlegend=False,
                        )
                        fig.update_traces(
                            hovertemplate=(
                                '<b>Latence :</b> %{x:.2f} '
                                'secondes<br><b>Nombre de réponses :</b> %{y}<extra></extra>'
                            ),
                            marker=dict(opacity=0.7, line=dict(color='black', width=1))
                        )
                        st.plotly_chart(fig, key="latency_chart")

                # Graphique de la répartition du coût total
                with st.container(border=True):
                    st.header(f"**💲 Coût total : {total_cost:.7f} €**")

                    if total_cost == 0:
                        st.info(
                            "Le graphique ne possède pas de données à afficher.",
                            icon=":material/info:"
                        )
                    else:
                        fig = px.pie(
                            names=conversation_names,
                            values=[
                                sum(
                                    message["metrics"].get("euro_cost", 0.0)
                                    for message in chat.get("messages", [])
                                    if message["role"] == "AI"
                                )
                                for chat in chats_to_analyze.values()
                            ],
                            color_discrete_sequence=px.colors.sequential.Greens[::-1]
                        )
                        fig.update_traces(
                            texttemplate='%{value:.7f}<br>%{percent}',
                            hovertemplate=(
                                '<b>%{label} :</b> %{value:.7f} € (%{percent})<extra></extra>'
                            )
                        )
                        st.plotly_chart(fig, key="cost_chart")

                # Graphique de la répartition de l'utilisation énergétique totale
                with st.container(border=True):
                    st.header(f"**⚡ Utilisation énergétique totale : {total_energy:.7f} kWh**")

                    if total_energy == 0:
                        st.info(
                            "Le graphique ne possède pas de données à afficher.",
                            icon=":material/info:"
                        )
                    else:
                        fig = px.pie(
                            names=conversation_names,
                            values=[
                                sum(
                                    message["metrics"].get("energy_usage", 0.0)
                                    for message in chat.get("messages", [])
                                    if message["role"] == "AI"
                                )
                                for chat in chats_to_analyze.values()
                            ],
                            color_discrete_sequence=px.colors.sequential.solar[::-1]
                        )
                        fig.update_traces(
                            texttemplate='%{value:.7f}<br>%{percent}',
                            hovertemplate=(
                                '<b>%{label} :</b> %{value:.7f} kWh (%{percent})<extra></extra>'
                            )
                        )
                        st.plotly_chart(fig, key="energy_chart")

                # Graphique de la répartition du potentiel de réchauffement global total
                with st.container(border=True):
                    st.header(
                        f"**🌡️ Potentiel de réchauffement global total : {total_gwp:.7f} kgCO2eq**"
                    )

                    if total_gwp == 0:
                        st.info(
                            "Le graphique ne possède pas de données à afficher.",
                            icon=":material/info:"
                        )
                    else:
                        fig = px.pie(
                            names=conversation_names,
                            values=[
                                sum(
                                    message["metrics"].get("gwp", 0.0)
                                    for message in chat.get("messages", [])
                                    if message["role"] == "AI"
                                )
                                for chat in chats_to_analyze.values()
                            ],
                            color_discrete_sequence=px.colors.sequential.Reds[::-1]
                        )
                        fig.update_traces(
                            texttemplate='%{value:.7f}<br>%{percent}',
                            hovertemplate=(
                                '<b>%{label} :</b> %{value:.7f} kgCO2eq (%{percent})<extra></extra>'
                            )
                        )
                        st.plotly_chart(fig, key="gwp_chart")

            # Affichage des KPIs
            else:
                # Affichage des statistiques
                cols = st.columns(4)
                with cols[0]:
                    with st.container(border=True):
                        st.write("**🗨️ Nombre total de messages envoyés**")
                        st.title(f"{total_user_messages}")
                with cols[1]:
                    with st.container(border=True):
                        st.write("**🛡️ Nombre total de messages bloqués**")
                        st.title(f"{total_blocked_messages}")
                with cols[2]:
                    with st.container(border=True):
                        st.write("**📄 Nombre total de documents importés**")
                        st.title(f"{total_documents_imported}")
                with cols[3]:
                    with st.container(border=True):
                        st.write("**🌐 Nombre total d'utilisations du mode internet**")
                        st.title(f"{total_internet_search}")
                cols = st.columns(4)
                with cols[0]:
                    with st.container(border=True):
                        st.write("**📶 Latence moyenne des réponses**")
                        st.title(f"{average_latency:.2f} secondes")
                with cols[1]:
                    with st.container(border=True):
                        st.write("**💲 Coût total**")
                        st.title(f"{total_cost:.7f} €")
                with cols[2]:
                    with st.container(border=True):
                        st.write("**⚡ Utilisation énergétique totale**")
                        st.title(f"{total_energy:.7f} kWh")
                with cols[3]:
                    with st.container(border=True):
                        st.write("**🌡️ Potentiel de réchauffement global total**")
                        st.title(f"{total_gwp:.7f} kgCO2eq")
        else:
            # Message d'information si aucune conversation n'a été sélectionnée
            st.info(
                "Veuillez sélectionner au moins une conversation pour afficher les statistiques.",
                icon=":material/info:",
            )
    else:
        # Message d'information si aucune conversation n'a été créée
        st.info(
            "Veuillez commencer une conversation pour afficher les statistiques.",
            icon=":material/info:",
        )
