import os
import sys
import sqlite3
import uuid
import torch
import pdfplumber
import tiktoken
import numpy as np
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModel
# Path adjustments for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from src.security.securite import LLMSecurityManager
from src.ml.promptClassifier import PromptClassifier
from src.llm.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray

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
    def __init__(self, db_path="llm_database.db", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)

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

    def calculate_embedding(self, documents: Union[str, List[str]]) -> NDArray[np.float32]:
        """
        Generates embeddings for a list of documents using SentenceTransformer model.

        Args:
            documents (Union[str, List[str]]): A string or a list of strings (documents) for which embeddings are to be generated.

        Returns:
            NDArray: A NumPy array containing the embeddings for each document.
        """
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        if isinstance(documents, str):
            documents = [documents]
        return model.encode(documents)

    def store_in_database(self, discussion_id: str, chunks: List[str]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                embedding = self.calculate_embedding(chunk)
                cursor.execute("""
                    INSERT INTO discussions (discussion_id, chunk_id, content, embedding)
                    VALUES (?, ?, ?, ?)
                """, (discussion_id, f"{discussion_id}_{i}", chunk, sqlite3.Binary(embedding.tobytes())))
            conn.commit()

    def process_pdf(self, pdf_path: str):
        discussion_id = str(uuid.uuid4())
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.split_into_chunks(text)
        self.store_in_database(discussion_id, chunks)
        print(f"PDF traité avec succès et stocké sous discussion_id : {discussion_id}")
