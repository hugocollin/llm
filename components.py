import time
import streamlit as st

# Fonction pour afficher le texte progressivement
def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

# Fonction pour obtenir le nom d'une nouvelle conversation
def get_new_chat_name():
    existing_numbers = [int(name.split(' ')[1]) for name in st.session_state["chats"].keys() if name.startswith("Chat ") and name.split(' ')[1].isdigit()]
    n = 1
    while n in existing_numbers:
        n += 1
    return f"Chat {n}"

# Fonction d'affichage de la barre latérale
def show_sidebar():
    # Initialisation des conversations
    if "chats" not in st.session_state:
        st.session_state["chats"] = {}

    with st.sidebar:
        # Message de bienvenue [TEMP]
        st.title("✨ SISE Classmate")
        
        if st.button("", icon=":material/bar_chart:"):
            st.toast("Fonctionnalité disponible ultérieurement", icon=":material/info:")

        header_cols = st.columns([3, 1])

        # Section des conversations
        with header_cols[0]:
            st.header("Conversations")

        # Bouton pour ajouter un chat
        with header_cols[1]:
            st.write("")
            if st.button("", icon=":material/add_comment:"):
                if len(st.session_state["chats"]) < 5:
                    new_chat_name = get_new_chat_name()
                    st.session_state["chats"][new_chat_name] = []
                else:
                    st.toast("Nombre maximal de conversations atteint, supprimez-en une pour en commencer une nouvelle", icon=":material/feedback:")

        # Sélecteur d'espaces de discussion
        if st.session_state["chats"]:
            selected_chat = None
            for chat_name in list(st.session_state["chats"].keys()):
                btn_cols = st.columns([3, 1])
                with btn_cols[0]:
                    if st.button(f":material/forum: {chat_name}"):
                        selected_chat = chat_name
                with btn_cols[1]:
                    if st.button("", icon=":material/delete:", key=f"del_chat_btn_{chat_name}"):
                        del st.session_state["chats"][chat_name]
                        if st.session_state.get('selected_chat') == chat_name:
                            st.session_state['selected_chat'] = next(iter(st.session_state["chats"]), None)
                        st.rerun()
            return selected_chat
        else:
            st.info("Pour commencer une nouvelle conversation, cliquez sur le bouton :material/add_comment:", icon=":material/info:")
            return None