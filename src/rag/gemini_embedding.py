import numpy as np
import google.generativeai as genai
from typing import List

class GoogleEmbedding:
    def __init__(self, api_key: str, model_name: str = "models/embedding-001"):
        genai.configure(api_key=api_key)
        self.model = model_name

    def embed_text(self, text: str) -> np.ndarray:
        """Génère un embedding pour un texte unique."""
        try:
            response = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = response.get("embedding", [])
            return np.array(embedding) if embedding else np.array([])
        except Exception as e:
            print(f"Erreur lors de l'embedding du texte : {e}")
            return np.array([])

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Génère des embeddings pour une liste de textes."""
        return [self.embed_text(text) for text in texts]

    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """Méthode requise par ChromaDB pour être compatible avec l'interface EmbeddingFunction."""
        return self.embed_batch(input)
