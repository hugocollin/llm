"""
Ce fichier contient les classes pipeline pour la gestion de la sécurité 
et le traitement des fichiers PDF.
"""

import os
import sys
import sqlite3
import uuid
from typing import List, Union
import numpy as np
from numpy.typing import NDArray
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from src.security.securite import LLMSecurityManager
from src.ml.prompt_classifier import PromptClassifier


# Ajout du répertoire parent au chemin de recherche des modules
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


class EnhancedLLMSecurityManager(LLMSecurityManager):
    """
    Gestionnaire de sécurité amélioré avec un classificateur 
    de prompts pour valider l'entrée utilisateur.
    """

    def __init__(
            self,
            role : str = "educational assistant",
            train_json_path : str = None,
            test_json_path : str = None,
            train_model : bool = False
        ):
        """
        Initialise le gestionnaire de sécurité avec un classificateur de prompts.
        Si train_model est True, un nouveau modèle est entraîné,
        sinon le modèle existant est chargé.

        Args:
            role (str, optional): Rôle attribué à l'assistant. Par défaut, "educational assistant".
            train_json_path (str, optional): Chemin vers le fichier JSON d'entraînement.
            test_json_path (str, optional): Chemin vers le fichier JSON de test.
            train_model (bool, optional): Indique s'il faut entraîner un nouveau modèle.
        """

        super().__init__(role)
        self.classifier = PromptClassifier()
        if train_model and train_json_path and test_json_path:
            self.classifier.load_train_and_test_data_from_json(train_json_path, test_json_path)
            self.classifier.train_and_evaluate()
            self.classifier.get_best_model()
            self.classifier.export_best_model("best_prompt_model.pkl")
        elif not train_model:
            self.classifier.load_model()

    def validate_input(self, user_input : str) -> bool:
        """
        Valide l'entrée utilisateur en utilisant un classificateur de prompts.

        Args:
            user_input (str): Entrée utilisateur à valider.

        Returns:
            bool: True si l'entrée est validée, False sinon.
        """
        cleaned_input = self.clean_input(user_input)
        is_valid, _ = super().validate_input(cleaned_input)
        if not is_valid:
            return is_valid
        prediction = self.classifier.predict_with_best_model([cleaned_input])[0]
        if prediction == 1:
            return False
        return True

class PDFPipeline:
    """
    Pipeline de traitement des fichiers PDF et de stockage dans une base de données.
    """

    def __init__(
            self,
            db_path : str = "llm_database.db",
            embedding_model : str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        """
        Initialise la pipeline de traitement des fichiers PDF
        et de stockage dans une base de données.

        Args:
            db_path (str, optional): 
            Chemin de la base de données SQLite. Par défaut, "llm_database.db".
            embedding_model (str, optional): Modèle utilisé pour générer des embeddings textuels. 
        """
        self.db_path = db_path
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM discussions")
            conn.commit()

    def split_into_chunks(self, text : str, chunk_size : int = 500) -> List[str]:
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

    def calculate_embedding(self, documents : Union[str, List[str]]) -> NDArray[np.float32]:
        """
        Génère des embeddings pour un ou plusieurs documents.

        Args:
            documents (Union[str, List[str]]): Une chaîne de 
            caractères ou une liste de chaînes de caractères à encoder.

        Returns:
            NDArray: Embeddings des documents.
        """
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        if isinstance(documents, str):
            documents = [documents]
        return model.encode(documents)

    def store_in_database(self, discussion_id : str, chunks : List[str]):
        """
        Stocke les segments d'un texte et leurs embeddings dans la base de données SQLite.

        Args:
            discussion_id (str): Identifiant unique de la discussion.
            chunks (List[str]): Liste des segments de texte.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                embedding = self.calculate_embedding(chunk)
                cursor.execute(
                    """
                    INSERT INTO discussions 
                    (discussion_id, chunk_id, content, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        discussion_id,
                        f"{discussion_id}_{i}",
                        chunk,
                        sqlite3.Binary(embedding.tobytes())
                    )
                )
            conn.commit()

    def process_txt(self, text : str) -> List[str]:
        """
        Traite un texte brut en le découpant en segments et en le stockant dans la base de données.

        Args:
            text (str): Texte brut à traiter.

        Returns:
            List[str]: Identifiants des discussions ajoutées à la base de données.
        """
        discussion_id = str(uuid.uuid4())
        chunks = self.split_into_chunks(text)
        self.store_in_database(discussion_id, chunks)
        print("[INFO] Le document a été ajouté avec succès à la base de données")

        return [discussion_id]
