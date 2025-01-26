import os
import sqlite3
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List
import uuid
import pdfplumber
import torch



class PDFPipeline:
    def __init__(self, db_path="discussions.db", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialise le pipeline avec une base de données et un modèle d'embedding.
        """
        self.db_path = db_path
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)

        # Initialiser la base de données
        self._init_database()

    def _init_database(self):
        """
        Initialise une base de données SQLite pour stocker les chunks et embeddings.
        """
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
        """
        Extrait le texte d'un fichier PDF.
        """
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Divise le texte en chunks de taille fixe.
        """
        tokens = text.split()
        chunks = [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
        return chunks

    def calculate_embedding(self, text: str) -> np.ndarray:
        """
        Calcule l'embedding pour un texte donné.
        """
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding

    def store_in_database(self, discussion_id: str, chunks: List[str]):
        """
        Stocke les chunks et leurs embeddings dans la base de données.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                embedding = self.calculate_embedding(chunk)
                cursor.execute("""
                    INSERT INTO discussions (discussion_id, chunk_id, content, embedding)
                    VALUES (?, ?, ?, ?)
                """, (discussion_id, f"{discussion_id}_{i}", chunk, embedding.tobytes()))
            conn.commit()

    def process_pdf(self, pdf_path: str):
        """
        Processus complet pour un fichier PDF : extraction, découpage, calcul d'embeddings et stockage.
        """
        discussion_id = str(uuid.uuid4())
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.split_into_chunks(text)
        self.store_in_database(discussion_id, chunks)
        print(f"PDF traité avec succès et stocké sous discussion_id : {discussion_id}")


# Exemple d'utilisation
if __name__ == "__main__":
    pipeline = PDFPipeline()
    pipeline.process_pdf("D:/M2 SISE/LLM/Présentation_Cours - Jour 1.pdf")