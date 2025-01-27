import time
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from logging import getLogger
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from src.db.vector_store import VectorStore
from src.rag.embedding_base import EmbeddingBase

@dataclass
class RAGMetrics:
    """Métriques détaillées pour une requête RAG."""
    latence: float = 0.0
    cout_euros: float = 0.0
    energie_kwh: float = 0.0
    empreinte_carbone_kg: float = 0.0
    tokens_entree: int = 0
    tokens_sortie: int = 0

def measure_latency(func):
    """Décorateur pour mesurer la latence des méthodes."""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        instance = args[0]  # L'instance de la classe est toujours le premier argument
        instance.last_latency = end_time - start_time

        if hasattr(instance, 'logger'):
            instance.logger.info(f"{func.__name__} : latence de {instance.last_latency:.4f} secondes")

        return result
    return wrapper

class RagRetriever:
    def __init__(self,
                 modele_embedding: EmbeddingBase,
                 vector_store: VectorStore,
                 nombre_resultats: int = 3,
                 prix_tokens: Optional[Dict[str, float]] = None,
                 zone_melange_electrique: str = "FRA",
                 logger: Optional[Any] = None):
        """
        Initialise un récupérateur RAG avancé avec métriques et logging.

        :param modele_embedding: Modèle d'embedding
        :param vector_store: Store de vecteurs
        :param nombre_resultats: Nombre de résultats à retourner
        :param prix_tokens: Coûts par million de tokens (input et output)
        :param zone_melange_electrique: Zone pour calcul carbone
        :param logger: Logger optionnel
        """
        self.modele_embedding = modele_embedding
        self.vector_store = vector_store
        self.nombre_resultats = nombre_resultats
        self.prix_tokens = prix_tokens or {"input": 1.95, "output": 5.85}
        self.zone_melange_electrique = zone_melange_electrique
        self.logger = logger or getLogger(__name__)
        self.last_latency = 0.0
        self.last_metrics = RAGMetrics()

    def _estimer_tokens(self, texte: str) -> int:
        """
        Estimation du nombre de tokens dans un texte.
        :param texte: Texte à tokenizer
        :return: Nombre estimé de tokens
        """
        return int(len(texte.split()) * 1.3)  # Approximation simple

    def _calculer_cout(self, tokens_entree: int, tokens_sortie: int) -> float:
        """
        Calcul du coût total en euros.
        :param tokens_entree: Nombre de tokens d'entrée
        :param tokens_sortie: Nombre de tokens de sortie
        :return: Coût en euros
        """
        return (
            (tokens_entree / 1_000_000) * self.prix_tokens["input"] +
            (tokens_sortie / 1_000_000) * self.prix_tokens["output"]
        )

    def _calculer_impact_environnemental(self, tokens: int) -> Dict[str, float]:
        """
        Calcul de l'impact énergétique et carbone.
        :param tokens: Nombre total de tokens
        :return: Dictionnaire contenant énergie et empreinte carbone
        """
        zones = {
            "FRA": (0.0005, 0.0002),  # France
            "USA": (0.0007, 0.0003),  # États-Unis
            "CHN": (0.001, 0.0005)    # Chine
        }
        energie_par, gwp_par = zones.get(self.zone_melange_electrique, (0.0005, 0.0002))
        return {
            "energie": (tokens / 1000) * energie_par,
            "gwp": (tokens / 1000) * gwp_par
        }

    def get_cosim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcul de la similarité cosinus entre deux vecteurs."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_top_similarity(self, embedding_query: np.ndarray, embedding_chunks: np.ndarray, corpus: List[str]) -> List[str]:
        """Retourne les documents les plus similaires à une requête."""
        cos_distances = [self.get_cosim(embedding_query, chunk) for chunk in embedding_chunks]
        indices_top = np.argsort(cos_distances)[-self.nombre_resultats:][::-1]
        return [corpus[i] for i in indices_top]

    def build_prompt(self, context: List[str], history: str, query: str) -> List[Dict[str, str]]:
        """Construit le prompt pour le modèle LLM."""
        context_text = "\n".join(context)
        return [
            {"role": "system", "content": "Votre prompt de système ici"},
            {"role": "system", "content": f"# Historique:\n{history}"},
            {"role": "system", "content": f"# Contexte:\n{context_text}"},
            {"role": "user", "content": f"# Question:\n{query}\n# Réponse:"}
        ]

    def call_model(self, prompt_dict: List[Dict[str, str]]) -> str:
        """Appelle le modèle avec un prompt donné."""
        return "Réponse générée par le modèle"

    @measure_latency
    def ajouter_documents(self, textes: List[str], metadonnees: Optional[List[Dict[str, Any]]] = None):
        """Ajoute des documents dans le store vectoriel."""
        embeddings = self.modele_embedding.embed_batch(textes)
        if len(textes) != len(embeddings):
            raise ValueError("Mismatch entre textes et embeddings")

        self.vector_store.collection.add(
            documents=textes,
            embeddings=embeddings,
            metadatas=metadonnees or [{}] * len(textes),
            ids=[f"doc_{hash(text)}" for text in textes]
        )

    @measure_latency
    def recuperer_documents(self, requete: str, filtre: Optional[Dict[str, Any]] = None) -> List[str]:
        """Récupère des documents pertinents à partir d'une requête."""
        embedding_requete = self.modele_embedding.embed_text(requete)
        resultats = self.vector_store.rechercher(
            requete=embedding_requete,
            nb_resultats=self.nombre_resultats,
            filtre_dict=filtre
        )

        documents = resultats.get("documents", [])
        metrics = RAGMetrics(
            latence=self.last_latency,
            tokens_entree=self._estimer_tokens(requete),
            tokens_sortie=sum(self._estimer_tokens(doc) for doc in documents),
        )
        impact = self._calculer_impact_environnemental(metrics.tokens_entree + metrics.tokens_sortie)
        metrics.energie_kwh = impact["energie"]
        metrics.empreinte_carbone_kg = impact["gwp"]
        metrics.cout_euros = self._calculer_cout(metrics.tokens_entree, metrics.tokens_sortie)
        self.last_metrics = metrics

        return documents
