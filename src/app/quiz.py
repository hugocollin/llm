import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


import streamlit as st
from src.rag.model_api import MultiModelLLM

class QuizApp:
    """
    Classe pour gérer la création et l'exécution d'un quiz généré par un LLM.
    """

    def __init__(self, llm):
        self.llm = llm

    def generate_quiz(self, topic, num_questions=5):
        """
        Génère un quiz avec des questions à choix multiples via le LLM.

        Args:
            topic (str): Sujet du quiz.
            num_questions (int): Nombre de questions à générer.

        Returns:
            list[dict]: Liste de questions avec les réponses et options.
        """
        prompt = (
            f"Tu es une intelligence artificielle spécialisée dans l'éducation. "
            f"Génère un quiz avec {num_questions} questions à choix multiples sur le sujet suivant : '{topic}'. "
            "Pour chaque question, fournis un dictionnaire JSON avec les clés suivantes : "
            "'question' (texte de la question), 'options' (liste de 4 options), "
            "'answer' (réponse correcte). Ne donne que le JSON en retour."
        )

        # Appel au LLM pour générer le quiz
        response = self.llm.generate(
            prompt=prompt,
            provider="mistral",
            model="mistral-large-latest",
            temperature=0.7,
            max_tokens=1000
        )

        # Conversion de la réponse en liste de questions
        try:
            quiz = eval(response["response"])
            return quiz
        except Exception as e:
            st.error("Erreur lors de la génération du quiz. Veuillez réessayer.")
            return []

    def display_quiz(self, quiz):
        """
        Affiche le quiz et collecte les réponses de l'utilisateur.

        Args:
            quiz (list[dict]): Liste de questions avec les réponses et options.

        Returns:
            dict: Résultats du quiz avec les réponses de l'utilisateur.
        """
        st.title("Quiz généré par l'IA 🎓")
        user_answers = {}

        for idx, question_data in enumerate(quiz):
            st.subheader(f"Question {idx + 1}")
            st.write(question_data["question"])
            options = question_data["options"]
            user_answers[idx] = st.radio(
                "Choisissez une réponse :",
                options=options,
                key=f"question_{idx}"
            )

        if st.button("Soumettre mes réponses"):
            return self.evaluate_quiz(quiz, user_answers)

    def evaluate_quiz(self, quiz, user_answers):
        """
        Évalue les réponses de l'utilisateur et affiche le score.

        Args:
            quiz (list[dict]): Liste de questions avec les réponses et options.
            user_answers (dict): Réponses de l'utilisateur.

        Returns:
            None
        """
        st.subheader("Résultats du Quiz")
        correct_answers = 0

        for idx, question_data in enumerate(quiz):
            st.write(f"**Question {idx + 1}:** {question_data['question']}")
            st.write(f"**Votre réponse :** {user_answers[idx]}")
            st.write(f"**Réponse correcte :** {question_data['answer']}")

            if user_answers[idx] == question_data["answer"]:
                st.success("✅ Correct")
                correct_answers += 1
            else:
                st.error("❌ Incorrect")

        st.write(f"### Score final : {correct_answers}/{len(quiz)}")

    def run(self):
        """
        Lance l'application Quiz.
        """
        st.sidebar.title("Quiz IA")
        st.sidebar.subheader("Paramètres")

        topic = st.sidebar.text_input("Sujet du Quiz", value="Les planètes du système solaire")
        num_questions = st.sidebar.slider("Nombre de questions", min_value=3, max_value=10, value=5)

        if st.sidebar.button("Générer le Quiz"):
            with st.spinner("Génération du Quiz..."):
                quiz = self.generate_quiz(topic, num_questions)

            if quiz:
                self.display_quiz(quiz)

def main():
    # Initialisation de l'API de l'IA
    llm = MultiModelLLM()

    # Création de l'instance de l'application Quiz
    quiz_app = QuizApp(llm)

    # Exécution de l'application Quiz
    quiz_app.run()

if __name__ == "__main__":

    main()