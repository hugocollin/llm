import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from logging import getLogger
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from src.db.vector_store import VectorStore
from src.rag.mistral_embedding import MistralEmbedding
from src.rag.gemini_embedding import GoogleEmbedding


@dataclass
class RAGMetrics:
    """Métriques détaillées pour une requête RAG."""
    latence: float = 0.0
    cout_euros: float = 0.0
    tokens_entree: int = 0
    tokens_sortie: int = 0


def measure_latency(func):
    """Décorateur pour mesurer la latence des méthodes."""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        self.last_latency = end_time - start_time

        if hasattr(self, 'logger'):
            self.logger.info(f"{func.__name__} : latence de {self.last_latency:.4f} secondes")

        return result
    return wrapper


class RagRetriever:
    def __init__(self,
                 modele_embedding: Union[MistralEmbedding, GoogleEmbedding],
                 vector_store: 'VectorStore',
                 nombre_resultats: int = 3,
                 prix_tokens: Optional[Dict[str, float]] = None,
                 logger: Optional[Any] = None,
                 template_prompt: Optional[str] = None):
        """
        Initialise un récupérateur RAG avancé avec métriques et logging.
        
        Args:
            modele_embedding: Modèle d'embedding à utiliser
            vector_store: Store vectoriel pour stocker les documents
            nombre_resultats: Nombre de résultats à retourner par requête
            prix_tokens: Dictionnaire des prix par token (input/output)
            logger: Logger personnalisé (optionnel)
            template_prompt: Template de prompt personnalisé (optionnel)
        """
        self.modele_embedding = modele_embedding
        self.vector_store = vector_store
        self.nombre_resultats = nombre_resultats
        self.prix_tokens = prix_tokens or {"input": 1.95, "output": 5.85}
        self.logger = logger or getLogger(__name__)
        self.last_latency = 0.0
        self.last_metrics = RAGMetrics()
        self.template_prompt = template_prompt or """
Contexte:
{context}

Question: {question}

Instructions:
- Utilisez uniquement les informations fournies dans le contexte ci-dessus
- Si vous ne pouvez pas répondre à partir du contexte, dites-le clairement
- Citez les passages pertinents du contexte pour justifier votre réponse
- Restez factuel et précis

Réponse:"""

    def _estimer_tokens(self, texte: str) -> int:
        """Estime le nombre de tokens dans un texte."""
        return int(len(texte.split()) * 1.3)

    def _calculer_cout(self, tokens_entree: int, tokens_sortie: int) -> float:
        """Calcule le coût total en euros basé sur le nombre de tokens utilisés."""
        return (
            (tokens_entree / 1_000_000) * self.prix_tokens["input"] +
            (tokens_sortie / 1_000_000) * self.prix_tokens["output"]
        )

    def _valider_et_aplatir_textes(self, textes: Union[str, List[Union[str, List[str]]]]) -> List[str]:
        """
        Valide et nettoie une liste de textes (supprime les vides, enlève les listes imbriquées).
        
        Args:
            textes: Texte unique ou liste de textes (avec potentiellement des listes imbriquées)
        Returns:
            Liste de chaînes de caractères valides
        """
        if isinstance(textes, str):
            textes = [textes]
        elif not isinstance(textes, list):
            raise TypeError("Le texte doit être une chaîne ou une liste de chaînes")

        return [t.strip() for sublist in textes for t in (sublist if isinstance(sublist, list) else [sublist]) 
                if isinstance(t, str) and t.strip()]

    @measure_latency
    def ajouter_documents(self, textes: Union[str, List[str]], 
                         metadonnees: Optional[List[Dict[str, Any]]] = None,
                         batch_size: int = 100) -> List[str]:
        """
        Ajoute des documents dans le store vectoriel avec gestion par lots.
        
        Args:
            textes: Documents à ajouter
            metadonnees: Métadonnées associées aux documents
            batch_size: Taille des lots pour le traitement
            
        Returns:
            Liste des IDs des documents ajoutés
        """
        textes_valides = self._valider_et_aplatir_textes(textes)
        
        if not textes_valides:
            raise ValueError("Aucun texte valide à ajouter après nettoyage.")

        documents_ids = []
        
        # Traitement par lots
        for i in range(0, len(textes_valides), batch_size):
            batch_textes = textes_valides[i:i + batch_size]
            
            embeddings = self.modele_embedding.embed_batch(batch_textes)

            if len(batch_textes) != len(embeddings):
                raise ValueError(f"Mismatch entre textes ({len(batch_textes)}) et embeddings ({len(embeddings)})")

            documents_finaux, embeddings_finaux, metadatas_finaux = [], [], []
            for j, (texte, emb) in enumerate(zip(batch_textes, embeddings)):
                if emb is not None and len(emb) > 0:
                    documents_finaux.append(texte)
                    embeddings_finaux.append(emb)
                    idx = i + j
                    metadatas_finaux.append(metadonnees[idx] if metadonnees and idx < len(metadonnees) 
                                          else {"source": "default"})
                else:
                    self.logger.warning(f"Embedding vide pour le texte : {texte}")

            if documents_finaux:
                batch_ids = [f"doc_{hash(text)}" for text in documents_finaux]
                self.vector_store.collection.add(
                    documents=documents_finaux,
                    embeddings=embeddings_finaux,
                    metadatas=metadatas_finaux,
                    ids=batch_ids
                )
                documents_ids.extend(batch_ids)
            
        if not documents_ids:
            raise ValueError("Aucun document valide n'a été ajouté.")
            
        return documents_ids

    @measure_latency
    def recuperer_documents(self, requete: str, 
                          filtre: Optional[Dict[str, Any]] = None,
                          seuil_similarite: float = 0.7) -> List[str]:
        """
        Récupère les documents pertinents pour une requête.
        
        Args:
            requete: Requête utilisateur
            filtre: Filtre à appliquer sur les métadonnées
            seuil_similarite: Seuil minimal de similarité (0-1)
            
        Returns:
            Liste des documents pertinents
        """
        if not isinstance(requete, str) or not requete.strip():
            raise ValueError("La requête doit être une chaîne de caractères non vide.")

        try:
            embedding_requete = self.modele_embedding.embed_text(requete)

            if embedding_requete is None or len(embedding_requete) == 0:
                raise ValueError("L'embedding généré pour la requête est vide.")

            resultats = self.vector_store.rechercher(
                requete=embedding_requete.tolist(),
                nb_resultats=self.nombre_resultats
            )

            # Filtrer par score de similarité
            documents = []
            scores = resultats.get("distances", [])
            for doc, score in zip(resultats.get("documents", []), scores):
                if score >= seuil_similarite:
                    documents.append(doc)

            if not documents:
                self.logger.warning("Aucun document ne dépasse le seuil de similarité.")
                return []

            return documents

        except Exception as e:
            raise RuntimeError(f"Erreur lors de la récupération des documents : {str(e)}")

    @measure_latency
    def build_prompt(self, question: str, documents: List[str]) -> str:
        """
        Construit le prompt final en combinant la question et les documents récupérés.
        
        Args:
            question: La question posée par l'utilisateur
            documents: Liste des documents récupérés
            
        Returns:
            Le prompt final formaté
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question doit être une chaîne non vide")
        
        if not documents:
            raise ValueError("La liste des documents ne peut pas être vide")
            
        try:
            context = "\n\n---\n\n".join(documents)
            
            self.last_metrics.tokens_entree = self._estimer_tokens(context + question)
            
            prompt_final = self.template_prompt.format(
                context=context,
                question=question
            )
            
            self.logger.debug(f"Taille du prompt (tokens estimés): {self.last_metrics.tokens_entree}")
            
            return prompt_final
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la construction du prompt : {str(e)}")

    @measure_latency
    def recuperer_et_construire(self, question: str, 
                              filtre: Optional[Dict[str, Any]] = None,
                              seuil_similarite: float = 0.7) -> str:
        """
        Méthode utilitaire qui combine la récupération des documents et la construction du prompt.
        
        Args:
            question: Question de l'utilisateur
            filtre: Filtre optionnel pour la recherche
            seuil_similarite: Seuil minimal de similarité
            
        Returns:
            Le prompt final prêt à être utilisé
        """
        try:
            documents = self.recuperer_documents(question, filtre, seuil_similarite)
            if not documents:
                return self.template_prompt.format(
                    context="Aucun document pertinent trouvé.",
                    question=question
                )
                
            return self.build_prompt(question, documents)
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la récupération et construction : {str(e)}")