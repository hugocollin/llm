"""
Ce fichier contient la classe RAG qui permet d'interagir
avec un modèle de langage pour la génération de réponses.
"""

import time
import functools
import sqlite3
import litellm
import numpy as np
import wikipedia
from ecologits import EcoLogits
from numpy.typing import NDArray


def measure_latency(func : callable) -> callable:
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction.

    Args:
        func (callable): La fonction à décorer.

    Returns:
        callable: La fonction décorée.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> float:
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time
        self.last_latency = latency
        return result

    return wrapper


class RAG:
    """
    Classe pour interagir avec un modèle de langage pour la génération de réponses.
    """

    def __init__(
        self,
        max_tokens : int,
        top_n : int,
    ):
        """
        Constructeur de la classe RAG.

        Args:
            max_tokens (int): Nombre maximum
            de tokens pour la génération de texte.
            top_n (int): Nombre de réponses
            similaires à retourner.
        """
        self.top_n = top_n
        self.max_tokens = max_tokens
        EcoLogits.init(providers="litellm", electricity_mix_zone="FRA")

    def get_cosim(self, a : NDArray[np.float32], b : NDArray[np.float32]) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs.

        Args:
            a (NDArray[np.float32]): Vecteur a.
            b (NDArray[np.float32]): Vecteur b.

        Returns:
            float: Similarité cosinus entre les vecteurs a et b.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_top_similarity(
        self,
        embedding_query : NDArray[np.float32],
        embedding_chunks : NDArray[np.float32],
        corpus : list[str],
    ) -> list[str]:
        """
        Retourne les documents les plus similaires à la requête.

        Args:
            embedding_query (NDArray[np.float32]): Embedding de la requête.
            embedding_chunks (NDArray[np.float32]): Embeddings des documents.
            corpus (list[str]): Liste des documents.

        Returns:
            list[str]: Liste des documents les plus similaires à la requête.
        """
        cos_dist_list = np.array(
            [
                self.get_cosim(embedding_query, embed_doc)
                for embed_doc in embedding_chunks
            ]
        )
        indices_of_max_values = np.argsort(cos_dist_list)[-self.top_n :][::-1]
        return [corpus[i] for i in indices_of_max_values]

    # def get_cours_embeddings(self) -> dict[str, NDArray[np.float32]]:
    #     """
    #     Récupère les embeddings des cours à partir de la base de données.

    #     Returns:
    #         dict[str, NDArray[np.float32]]: Dictionnaire des embeddings des cours
    #     """
    #     connection = sqlite3.connect('llm_database.db')
    #     cursor = connection.cursor()
    #     cursor.execute("SELECT nom_du_cours, embedding FROM cours")
    #     cours_data = cursor.fetchall()
    #     connection.close()
    #     return {nom: embedding for nom, embedding in cours_data}

    def get_documents_content(self, ressources : list[str]) -> str:
        """
        Récupère le contenu des documents à partir de la base de données
        en utilisant les id_conversation fournies.

        Args:
            ressources (list[str]): Liste des id_conversation.

        Returns:
            str: Contenu concaténé des documents.
        """
        contents = []
        db_path = "llm_database.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for discussion_id in ressources:
                cursor.execute(
                    "SELECT content FROM discussions WHERE discussion_id = ?",
                    (discussion_id,),
                )
                rows = cursor.fetchall()
                for row in rows:
                    contents.append(row[0])
        return "\n".join(contents)

    def fetch_wikipedia_data(self, query : str) -> str:
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

    def _get_price_query(
        self, model : str, input_tokens : int, output_tokens : int
    ) -> float:
        """
        Calcule le coût d'une requête en fonction du modèle LLM utilisé.

        Args:
            model (str): Modèle LLM utilisé.
            input_tokens (int): Nombre de tokens en entrée.
            output_tokens (int): Nombre de tokens en sortie.

        Returns:
            float: Coût de la requête.
        """
        pricing = {
            "ministral-8b-latest": {"input": 0.095, "output": 0.095},
            "mistral-large-latest": {"input": 1.92, "output": 5.75},
            "codestral-latest": {"input": 0.30, "output": 0.85},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
            "gemini-1.5-pro": {"input": 1.20, "output": 4.80},
        }
        if model not in pricing:
            raise ValueError(f"LLM {model} not found in pricing database.")
        cost_input = (input_tokens / 1_000_000) * pricing[model]["input"]
        cost_output = (output_tokens / 1_000_000) * pricing[model]["output"]
        return cost_input + cost_output

    def _get_energy_usage(self, response : litellm.ModelResponse) -> tuple[float, float]:
        """
        Calcule l'empreinte carbone et la consommation d'énergie d'une requête.

        Args:
            response (litellm.ModelResponse): Réponse du modèle LLM.

        Returns:
            tuple[float, float]: Consommation d'énergie et empreinte carbone.
        """
        energy_usage = getattr(
            response.impacts.energy.value, "min", response.impacts.energy.value
        )
        gwp = getattr(response.impacts.gwp.value, "min", response.impacts.gwp.value)
        return energy_usage, gwp

    def build_prompt(
        self,
        prompt_type : str,
        message : str = None,
        message_history : list[dict[str, str]] = None,
        ressources : list[str] = None,
        nb_questions : int = None,
    ) -> list[dict[str, str]]:
        """
        Construit les prompts pour les différentes tâches.

        Args:
            prompt_type (str): Type de prompt à construire.
            message (str): Message de l'utilisateur.
            message_history (list[dict[str, str]]): Historique de la conversation.
            ressources (list[str]): Liste des id_conversation.
            nb_questions (int): Nombre de questions pour le quiz.

        Returns:
            list[dict[str, str]]: Liste des prompts.
        """

        # Initialisation des prompts
        context_prompt = ""
        history_prompt = ""
        message_prompt = ""
        # courses_prompt = ""
        ressources_prompt = ""

        # Récupération de l'historique de la conversation
        if message_history:
            message_history = "\n".join(
                [
                    f"{message['role']}: {message['content']}"
                    for message in message_history
                    if "role" in message and "content" in message
                ]
            )
            message_history_formatted = (
                f"Voici l'historique de la conversation : {message_history}. "
            )
        else:
            message_history_formatted = ""

        # Construction du prompt pour la suggestion de questions
        if prompt_type == "suggestions":
            context_prompt = (
                "Tu es une intelligence artificielle spécialisée "
                "dans l'aide scolaire et éducative. "
            )
            message_prompt = (
                "Génère 5 questions courtes dans différentes matières sans les préciser, "
                "qu'un élève pourrait te poser sur une notion de cours. "
                "Répond uniquement en donnant les 5 questions sous forme de liste de tirets, "
                "sans explication supplémentaire."
            )

        # Construction du prompt pour la génération de noms de conversation
        elif prompt_type == "chat_name":
            context_prompt = (
                "Tu es une intelligence artificielle spécialisée "
                "dans la création de nom de conversation. "
                "En te basant sur le texte suivant, qui est le premier message de la conversation, "
                "propose un nom d'un maximum de 30 caractères pour cette conversation. "
                "Répond uniquement en donnant le nom de la conversation "
                "sans explication supplémentaire. "
            )
            message_prompt = (
                "Voici le premier message de la conversation "
                f"envoyé par l'utilisateur : {message}"
            )

        # Construction du prompt pour la génération de réponses à des messages
        elif prompt_type == "chat":
            # # Récupération des informations sur les cours
            # courses = self.get_cours_embeddings()

            # Récupération des informations des documents associés à la conversation
            if ressources:
                documents = self.get_documents_content(ressources)
            else:
                documents = "Pas de ressources supplémentaires fournies."

            context_prompt = (
                "Tu es une intelligence artificielle spécialisée "
                "dans l'aide scolaire et éducative. "
                "Tu réponds aux questions liées à l'école, aux cours, aux devoirs, aux quizz, "
                "aux examens, aux matières académiques , ainsi qu'à la culture générale. "
                "Tu peux fournir des explications, des résumés, des formules mathématiques, "
                "des exemples de code, des corrigés et des conseils pédagogiques. "
                "Si une question concerne une matière scolaire ou un sujet éducatif, "
                "réponds avec une réponse claire et détaillée. "
                "Si une formule mathématique est incluse dans ta réponse, "
                "entoure-la obligatoirement avec le symbole '$' pour qu'elle soit rendue en LaTeX. "
                "Si un exemple de code est inclus, assure-toi qu'il est bien "
                "formaté dans un bloc de code. "
                "Si la demande concerne une explication de réponse d'une question d'un quiz, "
                "réponds avec une explication détaillée. "
                "Si le message de l'utilisateur n'a aucun lien avec l'école, l'éducation, "
                "ou la culture générale, réponds uniquement et strictement par "
                "le mot 'Guardian' et rien d'autre. "
            )
            history_prompt = message_history_formatted
            message_prompt = f"Voici le message envoyé par l'utilisateur : {message} "
            # courses_prompt = (
            #     "Pour répondre au message suivant, nous te fournissons "
            #     f"les cours de l'Éducation nationale : {courses}"
            # )
            ressources_prompt = (
                "Pour répondre au message suivant, l'utilisateur a fourni des ressources "
                f"supplémentaires afin de te donner des informations sur le sujet : {documents}"
            )

        # Construction du prompt pour la génération de réponses à des messages avec le mode internet
        elif prompt_type == "internet_chat":
            # # Récupération des informations sur les cours
            # courses = self.get_cours_embeddings()

            # Récupération des informations sur Wikipedia
            wiki_summary = self.fetch_wikipedia_data(message)

            context_prompt = (
                "Tu es une intelligence artificielle spécialisée "
                "dans l'aide scolaire et éducative. "
                "Tu réponds aux questions liées à l'école, aux cours, aux devoirs, aux quizz, "
                "aux examens, aux matières académiques , ainsi qu'à la culture générale. "
                "Tu peux fournir des explications, des résumés, des formules mathématiques, "
                "des exemples de code, des corrigés et des conseils pédagogiques. "
                "Si une question concerne une matière scolaire ou un sujet éducatif, "
                "réponds avec une réponse claire et détaillée. "
                "Si une formule mathématique est incluse dans ta réponse, "
                "entoure-la obligatoirement avec le symbole '$' pour qu'elle soit rendue en LaTeX. "
                "Si un exemple de code est inclus, assure-toi qu'il est bien "
                "formaté dans un bloc de code. "
                "Si la demande concerne une explication de réponse d'une question d'un quiz, "
                "réponds avec une explication détaillée. "
                "Si le message de l'utilisateur n'a aucun lien avec l'école, l'éducation, "
                "ou la culture générale, réponds uniquement et strictement par "
                "le mot 'Guardian' et rien d'autre. "
            )
            history_prompt = message_history_formatted
            message_prompt = f"Voici le message envoyé par l'utilisateur : {message} "
            # courses_prompt = (
            #     "Pour répondre au message suivant, nous te fournissons "
            #     f"les cours de l'Éducation nationale : {courses}"
            # )
            ressources_prompt = (
                "Pour répondre au message suivant, nous te fournissons du contenu "
                "provenant d'un recherche sur Wikipedia "
                f"afin de te donner des informations sur le sujet : {wiki_summary}."
            )

        # Construction du prompt pour la génération de quiz
        elif prompt_type == "quizz":
            context_prompt = (
                "Tu es une intelligence artificielle spécialisée "
                "dans l'aide scolaire et éducative. "
                "Génère un quiz à choix multiples contenant "
                f"{nb_questions} questions sur le sujet donné. "
                "Retourne les questions sous forme d'un unique tableau JSON. "
                "Chaque question doit être un dictionnaire avec les clés suivantes : "
                "'question' (texte de la question), 'options' (liste de 4 options), "
                "'answer' (réponse correcte). "
                "Répond en envoyant uniquement et strictement le tableau JSON "
                "sans texte supplémentaire."
            )
            message_prompt = (
                "Les questions doivent être exclusivement et uniquement "
                f"sur les sujets évoqués dans la conversation. {message_history_formatted}"
            )
        return [
            {"role": "system", "content": context_prompt},
            {"role": "system", "content": history_prompt},
            {"role": "user", "content": message_prompt},
            # {"role": "user", "content": courses_prompt},
            {"role": "user", "content": ressources_prompt},
        ]

    def call_model(
        self,
        provider : str,
        model : str,
        temperature : float,
        prompt_dict : list[dict[str, str]],
    ) -> str:
        """
        Appelle le modèle de langage pour générer une réponse.

        Args:
            provider (str): Nom du fournisseur du modèle.
            model (str): Nom du modèle.
            temperature (float): Température pour l'échantillonnage.
            prompt_dict (list[dict[str, str]]): Liste des prompts.

        Returns:
            str: Réponse générée par le modèle.
        """
        response: litellm.ModelResponse = self._generate(
            provider, model, temperature, prompt_dict=prompt_dict
        )
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        euro_cost = self._get_price_query(model, input_tokens, output_tokens)
        energy_usage, gwp = self._get_energy_usage(response)
        response_text = str(response.choices[0].message.content)
        return {
            "response": response_text,
            "latency": getattr(self, "last_latency", 0),
            "euro_cost": euro_cost,
            "energy_usage": energy_usage,
            "gwp": gwp,
        }

    @measure_latency
    def _generate(
        self,
        provider : str,
        model : str,
        temperature : float,
        prompt_dict : list[dict[str, str]],
    ) -> litellm.ModelResponse:
        """
        Génère une réponse à partir des prompts donnés.

        Args:
            provider (str): Nom du fournisseur du modèle.
            model (str): Nom du modèle.
            temperature (float): Température pour l'échantillonnage.
            prompt_dict (list[dict[str, str]]): Liste des prompts.

        Returns:
            litellm.ModelResponse: Réponse générée par le modèle.
        """
        response = litellm.completion(
            model=f"{provider}/{model}",
            messages=prompt_dict,
            max_tokens=self.max_tokens,
            temperature=temperature,
        )
        return response

    def __call__(
        self,
        provider : str,
        model : str,
        temperature : float,
        prompt_type : str,
        message : str = None,
        message_history : list[dict[str, str]] = None,
        ressources : list[str] = None,
        nb_questions : int = None,
    ) -> str:
        """
        Appelle le modèle de langage pour générer une réponse.

        Args:
            provider (str): Nom du fournisseur du modèle.
            model (str): Nom du modèle.
            temperature (float): Température pour l'échantillonnage.
            prompt_type (str): Type de prompt à construire.
            message (str): Message de l'utilisateur.
            message_history (list[dict[str, str]]): Historique de la conversation.
            ressources (list[str]): Liste des id_conversation.
            nb_questions (int): Nombre de questions pour le quiz.

        Returns:
            str: Réponse générée par le modèle.
        """
        prompt = self.build_prompt(
            prompt_type, message, message_history, ressources, nb_questions
        )
        response = self.call_model(provider, model, temperature, prompt_dict=prompt)
        return response
