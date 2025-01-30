import os
import sys
import sqlite3
import uuid
import pdfplumber
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from src.security.securite import LLMSecurityManager
from src.ml.promptClassifier import PromptClassifier
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray

# --- Gestionnaire de sécurité ---
class EnhancedLLMSecurityManager(LLMSecurityManager):
    def __init__(self, user_input, role="educational assistant", train_json_path=None, test_json_path=None, train_model=False):
        """
        Initialise le gestionnaire de sécurité avec un classificateur de prompts.

        Args:
            user_input (str): Entrée utilisateur à valider.
            role (str, optional): Rôle attribué à l'assistant. Par défaut, "educational assistant".
            train_json_path (str, optional): Chemin vers le fichier JSON d'entraînement.
            test_json_path (str, optional): Chemin vers le fichier JSON de test.
            train_model (bool, optional): Indique s'il faut entraîner un nouveau modèle.

        If train_model is True and both JSON paths are provided, the classifier is trained, evaluated, and the best model is saved.
        Otherwise, an existing model is loaded.
        """
        
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
        """
        Valide l'entrée utilisateur en utilisant un classificateur de prompts.

        Returns:
            bool: True si l'entrée est valide, False sinon.

        L'entrée est d'abord nettoyée avant d'être validée par la classe parent.
        Ensuite, elle est évaluée par un modèle de classification.
        Si le modèle classe l'entrée comme suspecte (prédit 1), elle est rejetée.
        """
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
        """
        Initialise la pipeline de traitement des fichiers PDF et de stockage dans une base de données.

        Args:
            db_path (str, optional): Chemin de la base de données SQLite. Par défaut, "llm_database.db".
            embedding_model (str, optional): Modèle utilisé pour générer des embeddings textuels. 
        """
        self.db_path = db_path
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)


    def split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Divise un texte en segments de longueur définie.

        Args:
            text (str): Texte à diviser.
            chunk_size (int, optional): Nombre de tokens par segment. Par défaut, 500.

        Returns:
            List[str]: Liste des segments du texte.
        """
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
        """
        Stocke les segments d'un texte et leurs embeddings dans la base de données SQLite.

        Args:
            discussion_id (str): Identifiant unique de la discussion.
            chunks (List[str]): Liste des segments de texte.

        Chaque segment est associé à un ID unique et à un embedding avant d'être inséré dans la base de données.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                embedding = self.calculate_embedding(chunk)
                cursor.execute("""
                    INSERT INTO discussions (discussion_id, chunk_id, content, embedding)
                    VALUES (?, ?, ?, ?)
                    """, (discussion_id, f"{discussion_id}_{i}", chunk, sqlite3.Binary(embedding.tobytes())))
            conn.commit()
        

    def process_txt(self, text: str):
        """
        Traite un texte brut en le découpant en segments et en le stockant dans la base de données.

        Args:
            text (str): Texte brut à traiter.

        Un identifiant unique est généré pour la discussion, puis le texte est segmenté et stocké avec ses embeddings.
        """
        discussion_id = str(uuid.uuid4())
        chunks = self.split_into_chunks(text)
        self.store_in_database(discussion_id, chunks)
        print(f"PDF traité avec succès et stocké sous discussion_id : {discussion_id}")
