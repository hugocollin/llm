import numpy as np
import time
from mistralai import Mistral
from typing import List

class MistralEmbedding:
    def __init__(self, api_key: str, model_name: str = "mistral-embed"):
        if not api_key:
            raise ValueError("❌ La clé API Mistral est manquante. Vérifiez la configuration.")

        self.client = Mistral(api_key=api_key)
        self.model_name = model_name

    def embed_text(self, text: str) -> np.ndarray:
        """Génère un embedding pour un texte unique avec gestion des erreurs 429."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    inputs=[text]  # 🔥 Correction ici ("inputs" au lieu de "input")
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                if "429" in str(e):  # Trop de requêtes
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️ Trop de requêtes, attente de {wait_time} secondes...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Erreur lors de l'embedding du texte : {e}")
                    return np.array([])
        return np.array([])

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Génère des embeddings pour plusieurs textes avec pauses pour éviter les erreurs 429."""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            if embedding.size > 0:
                embeddings.append(embedding)
            else:
                embeddings.append(np.array([]))
            time.sleep(0.5)  # Pause entre les requêtes pour éviter les 429
        return embeddings
