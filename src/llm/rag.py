import time
import functools
import litellm
import numpy as np
import wikipedia
from ecologits import EcoLogits
from numpy.typing import NDArray

def measure_latency(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time
        self.last_latency = latency
        return result
    return wrapper

class RAG:
    def __init__(
        self,
        # bdd_chunks: BDDChunks,
        max_tokens: int,
        top_n: int,
    ) -> None:
        # self.bdd = bdd_chunks
        self.top_n = top_n
        self.max_tokens = max_tokens
        EcoLogits.init(providers="litellm", electricity_mix_zone="FRA")

    def get_cosim(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_top_similarity(
        self,
        embedding_query: NDArray[np.float32],
        embedding_chunks: NDArray[np.float32],
        corpus: list[str],
    ) -> list[str]:
        cos_dist_list = np.array(
            [
                self.get_cosim(embedding_query, embed_doc)
                for embed_doc in embedding_chunks
            ]
        )
        indices_of_max_values = np.argsort(cos_dist_list)[-self.top_n :][::-1]
        print(indices_of_max_values)
        return [corpus[i] for i in indices_of_max_values]
    
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
    
    def _get_price_query(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = {
            "ministral-8b-latest": {"input": 0.095, "output": 0.095},
            "mistral-large-latest": {"input": 1.92, "output": 5.75},
            "codestral-latest": {"input": 0.30, "output": 0.85},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
            "gemini-1.5-pro": {"input": 1.20, "output": 4.80}
        }
        if model not in pricing:
            raise ValueError(f"LLM {model} not found in pricing database.")
        cost_input = (input_tokens / 1_000_000) * pricing[model]["input"]
        cost_output = (output_tokens / 1_000_000) * pricing[model]["output"]
        return cost_input + cost_output
    
    def _get_energy_usage(self, response: litellm.ModelResponse) -> tuple[float, float]:
        energy_usage = getattr(response.impacts.energy.value, "min", response.impacts.energy.value)
        gwp = getattr(response.impacts.gwp.value, "min", response.impacts.gwp.value)
        return energy_usage, gwp

    # def build_prompt(
    #     self, context: list[str], history: str, query: str
    # ) -> list[dict[str, str]]:
    #     """
    #     Builds a prompt string for a conversational agent based on the given context and query.

    #     Args:
    #         context (str): The context information, typically extracted from books or other sources.
    #         query (str): The user's query or question.

    #     Returns:
    #         list[dict[str, str]]: The RAG prompt in the OpenAI format
    #     """
    #     context_joined = "\n".join(context)
    #     history_prompt = f"""
    #     # Historique de conversation:
    #     {history}
    #     """
    #     context_prompt = f"""
    #     Tu disposes de la section "Contexte" pour t'aider à répondre aux questions.
    #     # Contexte: 
    #     {context_joined}
    #     """
    #     query_prompt = f"""
    #     # Question:
    #     {query}

    #     # Réponse:
    #     """
    #     return [
    #         {"role": "system", "content": history_prompt},
    #         {"role": "system", "content": context_prompt},
    #         {"role": "user", "content": query_prompt},
    #     ]
    
    def build_prompt(self, type: str, message: str = None) -> list[dict[str, str]]:
        if type == "suggestions":
            # Construction du prompt personnalisé
            prompt = (
                "Tu es une intelligence artificielle spécialisée dans l'aide aux élèves à l'école. "
                "Génère 5 questions courtes dans différentes matières sans les préciser, "
                "qu'un élève pourrait te poser sur une notion de cours. "
                "Répond uniquement en donnant les 5 questions sous forme de liste de tirets, "
                "sans explication supplémentaire."
            )
        elif type == "chat_name":
            # Construction du prompt personnalisé
            prompt = (
                "Tu es une intelligence artificielle spécialisée "
                "dans la création de nom de conversation. "
                "En te basant sur le texte suivant, qui est le premier message de la conversation, "
                "propose un nom d'un maximum de 30 caractères pour cette conversation. "
                "Répond uniquement en donnant le nom de la conversation "
                "sans explication supplémentaire. "
                f"Voici le texte : {message}"
            )
        elif type == "chat":
            # Construction du prompt personnalisé
            prompt = (
                "Tu es une intelligence artificielle spécialisée dans l'aide "
                "et les réponses aux questions liées à l'éducation, l'école, "
                "les cours et la culture générale. Si un message reçu sort de ce cadre, "
                "tu réponds uniquement et strictement par le mot 'Guardian'. "
                #f"Voici l'histoire de la conversation : {history}. "
                f"Voici le message de l'utilisateur : {message}."
            )
        elif type == "internet_chat":
            # Récupération des informations sur Wikipedia
            wiki_summary = self.fetch_wikipedia_data(message)

            # Construction du prompt personnalisé
            prompt = (
                "Tu es une intelligence artificielle spécialisée dans l'aide "
                "et les réponses aux questions liées à l'éducation, l'école, "
                "les cours et la culture générale. Si un message reçu sort de ce cadre, "
                "tu réponds uniquement et strictement par le mot 'Guardian'. "
                #f"Voici l'historique de la conversation : {history}. "
                f"Voici le message de l'utilisateur : {message}. "
                "Pour répondre au message suivant, nous te fournissons du contenu "
                "provenant d'un recherche sur Wikipedia "
                f"afin de te donner des informations sur le sujet : {wiki_summary}."
            )
        return [
            {"role": "system", "content": prompt},
        ]

    def call_model(self, provider: str, model: str, temperature: float, prompt_dict: list[dict[str, str]]) -> str:
        response: litellm.ModelResponse = self._generate(provider, model, temperature, prompt_dict=prompt_dict)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        euro_cost = self._get_price_query(model, input_tokens, output_tokens)
        energy_usage, gwp = self._get_energy_usage(response)
        response_text = str(response.choices[0].message.content)
        return {
            "response": response_text,
            "latency": getattr(self, 'last_latency', 0),
            "euro_cost": euro_cost,
            "energy_usage": energy_usage,
            "gwp": gwp
        }
    
    @measure_latency
    def _generate(self, provider, model, temperature, prompt_dict: list[dict[str, str]]) -> litellm.ModelResponse:
        if provider == "gemini":
            model_name = model
        else:
            model_name = f"mistral/{model}"
        response = litellm.completion(
            model=model_name,
            messages=prompt_dict,
            max_tokens=self.max_tokens,
            temperature=temperature,
        )  # type: ignore

        return response

    def __call__(self, provider: str, model: str, temperature: float, type: str, message: str = None) -> str:
        # chunks = self.bdd.chroma_db.query(
        #     query_texts=[query],
        #     n_results=self.top_n,
        # )
        # chunks_list: list[str] = chunks["documents"][0]
        # prompt_rag = self.build_prompt(
        #     context=chunks_list, history=str(history), query=query
        # )
        prompt = self.build_prompt(type, message)
        response = self.call_model(provider, model, temperature, prompt_dict=prompt)
        return response
