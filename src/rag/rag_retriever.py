from typing import List, Dict, Any
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from src.db.vector_store import VectorStore
from src.rag.embedding_base import EmbeddingBase

class RagRetriever:
    def __init__(
        self,
        modele_embedding: EmbeddingBase,
        magasin_vecteurs: VectorStore,
        nombre_resultats: int = 3,
    ):
        """
        Initialise le récupérateur RAG.
        
        :param modele_embedding: Modèle d'embedding à utiliser
        :param magasin_vecteurs: Store de vecteurs pour stocker et rechercher les documents
        :param nombre_resultats: Nombre de résultats à retourner (par défaut 3)
        """
        self.modele_embedding = modele_embedding
        self.magasin_vecteurs = magasin_vecteurs
        self.nombre_resultats = nombre_resultats

    async def ajouter_documents(
        self, textes: List[str], metadonnees: List[Dict[str, Any]] = None
    ):
        """
        Ajoute des documents au magasin de vecteurs.
        
        :param textes: Liste des textes à ajouter
        :param metadonnees: Liste des métadonnées associées aux textes (optionnel)
        """
        # Vérification si la méthode existe dans l'embedder
        embeddings = await self.modele_embedding.embed_batch(textes)  # Ou générer_embeddings_lot si c'est le nom de la méthode
        self.magasin_vecteurs.ajouter_documents(textes, embeddings, metadonnees)

    async def recuperer_documents(
        self, requete: str, filtre: Dict[str, Any] = None
    ) -> List[str]:
        """
        Récupère les documents les plus pertinents pour une requête donnée.
        
        :param requete: Requête pour laquelle rechercher des documents
        :param filtre: Filtre à appliquer sur les métadonnées (optionnel)
        :return: Liste des documents les plus pertinents
        """
        # Générer l'embedding de la requête
        embedding_requete = await self.modele_embedding.embed_text(requete)  # Ou générer_embedding_texte si c'est le nom de la méthode
        resultats = self.magasin_vecteurs.rechercher(
            embedding_requete, nombre_resultats=self.nombre_resultats, filtre=filtre
        )
        # Retourne les documents
        return resultats.get("documents", [])[0] if resultats else []

    async def recuperer_documents_avec_scores(
        self, requete: str, filtre: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère les documents les plus pertinents avec leurs scores et métadonnées.
        
        :param requete: Requête pour laquelle rechercher des documents
        :param filtre: Filtre à appliquer sur les métadonnées (optionnel)
        :return: Liste de dictionnaires contenant les documents, scores et métadonnées
        """
        # Générer l'embedding de la requête
        embedding_requete = await self.modele_embedding.embed_text(requete)
        resultats = self.magasin_vecteurs.rechercher(
            embedding_requete, nombre_resultats=self.nombre_resultats, filtre=filtre
        )

        if resultats:
            return [
                {"document": doc, "score": score, "metadonnees": meta}
                for doc, score, meta in zip(
                    resultats["documents"][0],
                    resultats["distances"][0],
                    resultats["metadonnees"][0],
                )
            ]
        return []
