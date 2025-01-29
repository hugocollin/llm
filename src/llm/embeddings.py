import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self):
        """
        Initialise l'instance avec SentenceTransformer.
        """
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Génère un embedding pour un texte unique."""
        embedding = self.model.encode(text)
        return np.array(embedding)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Génère des embeddings pour une liste de textes."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embedding for embedding in embeddings]
    
    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """Méthode pour compatibilité avec d'autres interfaces."""
        return self.embed_batch(input)
    
# class MistralEmbedding:
#     def __init__(self, api_key: str, model_name: str = "mistral-embed"):
#         if not api_key:
#             raise ValueError("La clé API Mistral est manquante. Vérifiez la configuration.")

#         self.client = Mistral(api_key=api_key)
#         self.model_name = model_name

#     def embed_text(self, text: str) -> np.ndarray:
#         """Génère un embedding pour un texte unique avec gestion des erreurs 429."""
#         max_retries = 5
#         for attempt in range(max_retries):
#             try:
#                 response = self.client.embeddings.create(
#                     model=self.model_name,
#                     inputs=[text]  
#                 )
#                 return np.array(response.data[0].embedding)
#             except Exception as e:
#                 if "429" in str(e): 
#                     wait_time = 2 ** attempt  
#                     time.sleep(wait_time)
#                 else:
#                     print(f"Erreur lors de l'embedding du texte : {e}")
#                     return np.array([])
#         return np.array([])

#     def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
#         """Génère des embeddings pour plusieurs textes avec pauses pour éviter les erreurs 429."""
#         embeddings = []
#         for text in texts:
#             embedding = self.embed_text(text)
#             if embedding.size > 0:
#                 embeddings.append(embedding)
#             else:
#                 embeddings.append(np.array([]))
#             time.sleep(0.5) 
#         return embeddings

# class GoogleEmbedding:
#     def __init__(self, api_key: str, model_name: str = "models/embedding-001"):
#         genai.configure(api_key=api_key)
#         self.model = model_name

#     def embed_text(self, text: str) -> np.ndarray:
#         """Génère un embedding pour un texte unique."""
#         try:
#             response = genai.embed_content(
#                 model=self.model,
#                 content=text,
#                 task_type="retrieval_document"
#             )
#             embedding = response.get("embedding", [])
#             return np.array(embedding) if embedding else np.array([])
#         except Exception as e:
#             print(f"Erreur lors de l'embedding du texte : {e}")
#             return np.array([])

#     def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
#         """Génère des embeddings pour une liste de textes."""
#         return [self.embed_text(text) for text in texts]

#     def __call__(self, input: List[str]) -> List[np.ndarray]:
#         """Méthode requise par ChromaDB pour être compatible avec l'interface EmbeddingFunction."""
#         return self.embed_batch(input)