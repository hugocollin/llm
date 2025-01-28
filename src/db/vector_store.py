import os
import sys
import uuid
import fitz
import tiktoken
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# Adjust Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from src.rag.embedding_base import EmbeddingBase
from src.rag.mistral_embedding import MistralEmbedding
from src.rag.gemini_embedding import GoogleEmbedding


class VectorStore:
    def __init__(
        self,
        chemin_persistance: str = "../db/ChromaDB",
        modele_embedding: str = "gemini-1.5-flash",
        nom_collection: str = "collection_par_defaut",
        batch_size: int = 100,
        gemini_api_key: str = None,
        mistral_api_key: str = None
    ) -> None:
        if not isinstance(chemin_persistance, str) or not chemin_persistance.strip():
            raise ValueError("Le chemin de persistance doit être une chaîne non vide.")
        if batch_size <= 0:
            raise ValueError("batch_size doit être supérieur à 0.")
        if not gemini_api_key or not mistral_api_key:
            raise ValueError("Les clés API GEMINI et MISTRAL doivent être fournies.")

        self.batch_size = batch_size
        self.chemin_persistance = chemin_persistance
        self.gemini_api_key = gemini_api_key
        self.mistral_api_key = mistral_api_key
        
        os.makedirs(chemin_persistance, exist_ok=True)

        try:
            self.encodeur = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            raise RuntimeError(f"Erreur d'initialisation de Tiktoken : {str(e)}")

        try:
            self.client = chromadb.PersistentClient(
                path=chemin_persistance,
                settings=Settings(anonymized_telemetry=False)
            )
            self.fonction_embedding = self._get_embedding_function(modele_embedding)
            self.collection = self._creer_collection(nom_collection)
        except Exception as e:
            raise RuntimeError(f"Erreur d'initialisation de ChromaDB : {str(e)}")

    def _get_embedding_function(self, model_name: str):
        if model_name.startswith("gemini"):
            return GoogleEmbedding(api_key=self.gemini_api_key)
        elif model_name.startswith("mistral"):
            return MistralEmbedding(api_key=self.mistral_api_key)
        else:
            raise ValueError(f"Modèle d'embedding non pris en charge : {model_name}")

    def _creer_collection(self, nom: str) -> chromadb.Collection:
        # Normaliser le nom pour respecter les contraintes Chroma
        nom = "a" + nom[:50].strip().replace(" ", "_") + "a"
        return self.client.get_or_create_collection(
            name=nom,
            embedding_function=self.fonction_embedding,
            metadata={"hnsw:space": "cosine"}
        )

    def lire_pdf(self, chemin_fichier: str) -> str:
        if not os.path.exists(chemin_fichier):
            raise FileNotFoundError(f"Le fichier {chemin_fichier} n'existe pas")
        try:
            doc = fitz.open(chemin_fichier)
            return "".join(page.get_text() for page in doc)
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du PDF: {str(e)}")

    def decouper_texte(self, texte: str, taille_chunk: int = 500) -> List[str]:
        if not texte.strip():
            raise ValueError("Le texte ne peut pas être vide")
        tokens = self.encodeur.encode(texte)
        return [
            self.encodeur.decode(tokens[i:i + taille_chunk])
            for i in range(0, len(tokens), taille_chunk)
        ]

    def ajouter_documents(
        self,
        textes: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Ajoute les documents en lots à la collection ChromaDB,
        avec ou sans embeddings pré-calculés.
        """
        if not textes:
            raise ValueError("La liste des textes ne peut pas être vide")
        
        document_ids = [f"doc_{hash(text)}" for text in textes]

        try:
            # Nous utilisons tqdm pour afficher la progression
            for i in tqdm(range(0, len(textes), self.batch_size)):
                batch = {
                    "documents": textes[i:i + self.batch_size],
                    "ids": document_ids[i:i + self.batch_size]
                }
                
                if embeddings:
                    batch["embeddings"] = embeddings[i:i + self.batch_size]
                if metadatas:
                    batch["metadatas"] = metadatas[i:i + self.batch_size]
                    
                self.collection.add(**batch)
                
            return document_ids
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'ajout des documents: {str(e)}")

    def rechercher(self, requete: Union[str, List[float]], nb_resultats: int = 3) -> Dict[str, List]:
        """
        Recherche des documents similaires dans la collection.
        
        :param requete: Soit une chaîne de texte, soit un vecteur d'embedding
        :param nb_resultats: Nombre de résultats à retourner
        :return: Dictionnaire contenant les documents et leurs scores
        """
        try:
            if isinstance(requete, str):
                # Si c'est une chaîne, utiliser la méthode query standard
                resultats = self.collection.query(
                    query_texts=[requete],
                    n_results=nb_resultats
                )
            elif isinstance(requete, list):
                # Si c'est un vecteur d'embedding, utiliser query_embeddings
                resultats = self.collection.query(
                    query_embeddings=[requete],
                    n_results=nb_resultats
                )
            else:
                raise ValueError("La requête doit être une chaîne ou un vecteur d'embedding")
            
            return {
                "documents": resultats["documents"][0] if resultats["documents"] else [],
                "distances": resultats["distances"][0] if resultats["distances"] else []
            }
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la recherche: {str(e)}")
