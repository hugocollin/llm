import functools
import time
from typing import List, Dict, Any, Optional, Union
import sys
import os
import numpy as np
from dataclasses import dataclass, field
from logging import getLogger
import asyncio
from logging import basicConfig
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from src.db.vector_store import VectorStore
from embedding_base import EmbeddingBase
from mistral_embedding import MistralEmbedding
from gemini_embedding import GoogleEmbedding


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
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = await func(self, *args, **kwargs)
        end_time = time.time()
        self.last_latency = end_time - start_time
        
        # Log de la latence si un logger est disponible
        if hasattr(self, 'logger'):
            self.logger.info(f"Méthode {func.__name__} : latence de {self.last_latency:.4f} secondes")
        
        return result
    return wrapper

class RagRetriever:
    def __init__(
        self,
        modele_embedding: EmbeddingBase,
        magasin_vecteurs: VectorStore,
        nombre_resultats: int = 3,
        prix_tokens: Dict[str, float] = None,
        zone_melange_electrique: str = "FRA",
        logger: Optional[Any] = None
    ):
        """
        Initialise un récupérateur RAG avancé avec logging et métriques.
        
        :param modele_embedding: Modèle d'embedding
        :param magasin_vecteurs: Store de vecteurs
        :param nombre_resultats: Nombre de résultats à retourner
        :param prix_tokens: Coûts par million de tokens
        :param zone_melange_electrique: Zone pour calcul carbone
        :param logger: Logger optionnel
        """
        self.modele_embedding = modele_embedding
        self.magasin_vecteurs = magasin_vecteurs
        self.nombre_resultats = nombre_resultats
        self.prix_tokens = prix_tokens or {"input": 1.95, "output": 5.85}
        self.zone_melange_electrique = zone_melange_electrique
        self.logger = logger or getLogger(__name__)
        self.last_latency = 0.0  # Stockage de la dernière latence

    def _estimer_tokens(self, texte: str) -> int:
        """
        Estimation précise du nombre de tokens.
        
        :param texte: Texte à tokenizer
        :return: Nombre estimé de tokens
        """
        return len(texte.split()) * 1.3  # Approximation améliorée

    def _calculer_cout(self, tokens_entree: int, tokens_sortie: int) -> float:
        """
        Calcul précis du coût en euros.
        
        :param tokens_entree: Nombre de tokens d'entrée
        :param tokens_sortie: Nombre de tokens de sortie
        :return: Coût total
        """
        return (
            (tokens_entree / 1_000_000) * self.prix_tokens["input"] +
            (tokens_sortie / 1_000_000) * self.prix_tokens["output"]
        )

    def _calculer_impact_environnemental(self, tokens: int) -> Dict[str, float]:
        """
        Calcul de l'impact environnemental selon la zone.
        
        :param tokens: Nombre de tokens
        :return: Impact énergétique et carbone
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

    @measure_latency
    async def ajouter_documents(
        self, 
        textes: List[str], 
        metadonnees: Optional[List[Dict[str, Any]]] = None
    ):
        try:
            embeddings = await self.modele_embedding.embed_batch(textes)
            
            # Ajustement pour correspondre à votre VectorStore
            if not metadonnees:
                metadonnees = [{} for _ in textes]
            
            self.magasin_vecteurs.ajouter_documents(textes, metadonnees)
            
            self.logger.info(f"Ajouté {len(textes)} documents")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de documents : {e}")
            raise

    @measure_latency
    async def ajouter_documents(
        self, 
        textes: List[str], 
        metadonnees: Optional[List[Dict[str, Any]]] = None
    ):
        try:
            embeddings = await self.modele_embedding.embed_batch(textes)
            
            # Vérification ou initialisation des métadonnées
            if not metadonnees:
                metadonnees = [{} for _ in textes]
            
            self.magasin_vecteurs.collection.add(
                documents=textes,
                embeddings=embeddings,
                metadatas=metadonnees,
                ids=[f"doc_{i}" for i in range(len(textes))]
            )
            self.logger.info(f"Ajouté {len(textes)} documents avec embeddings.")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de documents : {e}")
            raise

    @measure_latency
    async def recuperer_documents(
        self, 
        requete: str, 
        filtre: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[List, float, RAGMetrics]]:
        metrics = RAGMetrics()

        try:
            embedding_requete = await self.modele_embedding.embed_text(requete)
            resultats = self.magasin_vecteurs.rechercher(
                embedding_requete, 
                nb_resultats=self.nombre_resultats, 
                filtre_dict=filtre
            )

            metrics.latence = self.last_latency
            metrics.tokens_entree = self._estimer_tokens(requete)
            metrics.tokens_sortie = sum(
                self._estimer_tokens(doc) for doc in resultats.get("documents", [])
            )

            impact = self._calculer_impact_environnemental(
                metrics.tokens_entree + metrics.tokens_sortie
            )
            metrics.energie_kwh = impact["energie"]
            metrics.empreinte_carbone_kg = impact["gwp"]
            metrics.cout_euros = self._calculer_cout(metrics.tokens_entree, metrics.tokens_sortie)

            return {
                "documents": resultats.get("documents", []),
                "scores": resultats.get("distances", []),
                "metriques": metrics
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération : {e}")
            raise

    def get_last_latency(self) -> float:
        """
        Retourne la dernière latence mesurée.
        
        :return: Dernière latence en secondes
        """
        return self.last_latency
    