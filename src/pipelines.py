import os
import sys
import sqlite3
import uuid
import torch
from typing import List, Dict, Optional, Any, Union
from transformers import AutoTokenizer, AutoModel
import pdfplumber
import chromadb
from chromadb.config import Settings
import tiktoken
from src.security.securite import LLMSecurityManager
from src.ml.promptClassifier import PromptClassifier
# from rag.embedding_base import EmbeddingBase
# from rag.mistral_embedding import MistralEmbedding
# from rag.gemini_embedding import GoogleEmbedding

# Path adjustments for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

# --- Gestionnaire de sécurité ---
class EnhancedLLMSecurityManager(LLMSecurityManager):
    def __init__(self, user_input, role="educational assistant", train_json_path=None, test_json_path=None, train_model=False):
        super().__init__(role)
        self.user_input = user_input
        self.classifier = PromptClassifier()
        if train_model and train_json_path and test_json_path:
            self.classifier.load_train_and_test_data_from_json(train_json_path, test_json_path)
            self.classifier.train_and_evaluate()
            self.classifier.get_best_model()
            self.classifier.export_best_model("best_prompt_model.pkl")
        elif not train_model:
            self.classifier.load_model()

    def validate_input(self):
        cleaned_input = self.clean_input(self.user_input)
        is_valid, _ = super().validate_input(cleaned_input)
        if not is_valid:
            return is_valid
        prediction = self.classifier.predict_with_best_model([cleaned_input])[0]
        if prediction == 1:
            return False
        return True

# --- Pipeline PDF et Base de données ---
class PDFPipeline:
    def __init__(self, db_path="discussions.db", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discussions (
                    discussion_id TEXT,
                    chunk_id TEXT,
                    content TEXT,
                    embedding BLOB
                )
            """)
            conn.commit()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        tokens = text.split()
        chunks = [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
        return chunks

    def calculate_embedding(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding

    def store_in_database(self, discussion_id: str, chunks: List[str]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                embedding = self.calculate_embedding(chunk)
                cursor.execute("""
                    INSERT INTO discussions (discussion_id, chunk_id, content, embedding)
                    VALUES (?, ?, ?, ?)
                """, (discussion_id, f"{discussion_id}_{i}", chunk, embedding.numpy().tobytes()))
            conn.commit()

    def process_pdf(self, pdf_path: str):
        discussion_id = str(uuid.uuid4())
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.split_into_chunks(text)
        self.store_in_database(discussion_id, chunks)
        print(f"PDF traité avec succès et stocké sous discussion_id : {discussion_id}")

# --- Gestion des Embeddings ---
class VectorStore:
    def __init__(self, chemin_persistance: str = "../db/ChromaDB", modele_embedding: str = "gemini-1.5-flash", nom_collection: str = "collection_par_defaut", batch_size: int = 100, gemini_api_key: str = None, mistral_api_key: str = None) -> None:
        self.batch_size = batch_size
        self.chemin_persistance = chemin_persistance
        self.gemini_api_key = gemini_api_key
        self.mistral_api_key = mistral_api_key
        os.makedirs(chemin_persistance, exist_ok=True)
        self.encodeur = tiktoken.get_encoding("cl100k_base")
        self.client = chromadb.PersistentClient(path=chemin_persistance, settings=Settings(anonymized_telemetry=False))
        self.fonction_embedding = self._get_embedding_function(modele_embedding)
        self.collection = self._creer_collection(nom_collection)

    def _get_embedding_function(self, model_name: str):
        if model_name.startswith("gemini"):
            return GoogleEmbedding(api_key=self.gemini_api_key)
        elif model_name.startswith("mistral"):
            return MistralEmbedding(api_key=self.mistral_api_key)
        else:
            raise ValueError(f"Modèle d'embedding non pris en charge : {model_name}")

    def _creer_collection(self, nom: str) -> chromadb.Collection:
        nom = "a" + nom[:50].strip().replace(" ", "_") + "a"
        return self.client.get_or_create_collection(name=nom, embedding_function=self.fonction_embedding, metadata={"hnsw:space": "cosine"})

    def rechercher(self, requete: Union[str, List[float]], nb_resultats: int = 3) -> Dict[str, List]:
        if isinstance(requete, str):
            resultats = self.collection.query(query_texts=[requete], n_results=nb_resultats)
        elif isinstance(requete, list):
            resultats = self.collection.query(query_embeddings=[requete], n_results=nb_resultats)
        else:
            raise ValueError("La requête doit être une chaîne ou un vecteur d'embedding")
        return {"documents": resultats["documents"][0] if resultats["documents"] else [], "distances": resultats["distances"][0] if resultats["distances"] else []}







# --- Exemple d'utilisation combinée ---
if __name__ == "__main__":
    user_input = "Comment afficher Hello World en Python ?"
    train_json_path = os.path.join("ml", "guardrail_dataset_train.json")
    test_json_path = os.path.join("ml", "guardrail_dataset_test.json")
    
    # Gestionnaire de sécurité
    security_manager = EnhancedLLMSecurityManager(user_input=user_input, role="educational assistant", train_json_path=train_json_path, test_json_path=test_json_path, train_model=False)
    if not security_manager.validate_input():
        print("L'entrée utilisateur est invalide!")
    else:
        print("Entrée validée!")

        # Traitement du PDF
        pdf_pipeline = PDFPipeline()
        pdf_pipeline.process_pdf("D:/M2 SISE/LLM/Présentation_Cours - Jour 1.pdf")
        
        # Ajout des données à la base de données vectorielle
        vector_store = VectorStore(chemin_persistance="../db/ChromaDB", modele_embedding="gemini-1.5-flash", nom_collection="collection_par_defaut", batch_size=100, gemini_api_key="your_api_key", mistral_api_key="your_api_key")
        # Utiliser vector_store pour ajouter et rechercher des documents après avoir stocké les données
