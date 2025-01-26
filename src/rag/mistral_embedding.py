import numpy as np
from typing import List
from mistralai import Mistral
from embedding_base import EmbeddingBase


class MistralEmbedding:
    def __init__(self, api_key: str, model_name: str = "mistral-embed"):
        """Initialise le client Mistral avec la clé API et le modèle."""
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name

    def embed_text(self, text: str) -> np.ndarray:
        """Génère un embedding pour un texte unique."""
        try:
            # Appelle l'API Mistral pour générer un embedding
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=[text]
            )
            # Accède à l'attribut `data` de la réponse
            embedding = response.data[0].embedding
            return np.array(embedding)
        except Exception as e:
            print(f"Erreur lors de l'embedding du texte : {e}")
            return np.array([])

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Génère des embeddings pour un lot de textes."""
        try:
            # Appelle l'API Mistral pour générer des embeddings
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=texts
            )
            # Accède à l'attribut `data` de la réponse pour récupérer les embeddings
            embeddings = [np.array(item.embedding) for item in response.data]
            return embeddings
        except Exception as e:
            print(f"Erreur lors de l'embedding du batch : {e}")
            return [np.array([]) for _ in texts]
