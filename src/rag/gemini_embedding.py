import numpy as np
import asyncio
import os
import sys
import google.generativeai as genai
from typing import List
from abc import ABC, abstractmethod

# Ajouter le chemin du répertoire parent au chemin système
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

# Importer les classes de base
from llm.embedding.embedding_base import EmbeddingBase


class GoogleEmbedding(EmbeddingBase):
    def __init__(self, api_key: str, model_name: str = "models/embedding-001"):
        genai.configure(api_key=api_key)
        self.model = model_name

    async def embed_text(self, text: str) -> np.ndarray:
        """Génère un embedding pour un texte unique."""
        try:
            # Appel à l'API pour un seul texte
            response = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = response.get("embedding", [])
            if not embedding:
                print(f"Aucun embedding généré pour le texte : {text}")
            return np.array(embedding)
        except Exception as e:
            print(f"Erreur lors de l'embedding du texte : {e}")
            return np.array([])

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Génère des embeddings pour un lot de textes."""
        try:
            embeddings = []
            for text in texts:
                # Appel à l'API pour chaque texte individuellement
                response = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embedding = response.get("embedding", [])
                if not embedding:
                    print(f"Aucun embedding généré pour le texte : {text}")
                embeddings.append(np.array(embedding))
            return embeddings
        except Exception as e:
            print(f"Erreur lors de l'embedding du batch : {e}")
            return [np.array([]) for _ in texts]
