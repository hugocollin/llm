import os
import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader


class EnhancedLLMSecurityManager:
    """
    Gère la validation et la sécurisation des prompts utilisateur via un ensemble de règles.

    :param allowed_words: Liste des mots autorisés dans le prompt utilisateur.
    """
    
    def __init__(self, allowed_words: List[str]):
        self.allowed_words = set(allowed_words)

    def is_valid_prompt(self, prompt: str) -> bool:
        """
        Vérifie si un prompt utilisateur est valide en fonction des mots autorisés.

        :param prompt: Chaîne de caractères à valider.
        :return: Booléen indiquant si le prompt est valide.
        """
        prompt_words = set(prompt.lower().split())
        invalid_words = prompt_words - self.allowed_words
        return len(invalid_words) == 0

    def sanitize_prompt(self, prompt: str) -> str:
        """
        Nettoie un prompt utilisateur en supprimant les mots non autorisés.

        :param prompt: Chaîne de caractères à nettoyer.
        :return: Prompt nettoyé.
        """
        prompt_words = prompt.split()
        sanitized = " ".join(word for word in prompt_words if word.lower() in self.allowed_words)
        return sanitized


class PDFPipeline:
    """
    Gère le traitement, le découpage et l'indexation des documents PDF.

    :param db_path: Chemin du répertoire pour stocker la base de données des vecteurs.
    :param embedding_model: Modèle utilisé pour générer les embeddings.
    """
    
    def __init__(self, db_path: str, embedding_model: str = "text-embedding-ada-002"):
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Charge un document PDF en mémoire.

        :param pdf_path: Chemin du fichier PDF.
        :return: Liste d'objets Document représentant les pages du PDF.
        """
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    def split_text(self, documents: List[Document], chunk_size: int = 500, overlap: int = 50) -> List[Document]:
        """
        Découpe le contenu des documents en segments plus petits.

        :param documents: Liste d'objets Document à découper.
        :param chunk_size: Taille maximale de chaque segment en caractères.
        :param overlap: Taille de chevauchement entre les segments.
        :return: Liste de segments découpés.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_documents(documents)

    def store_embeddings(self, documents: List[Document], collection_name: str):
        """
        Génère et stocke les embeddings des documents dans une base vectorielle.

        :param documents: Liste d'objets Document à indexer.
        :param collection_name: Nom de la collection dans la base vectorielle.
        """
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )
        self.vector_store.add_documents(documents)
        self.vector_store.persist()

    def query_embeddings(self, query: str, top_k: int = 5) -> List[str]:
        """
        Interroge la base vectorielle pour trouver les documents les plus similaires.

        :param query: Requête utilisateur.
        :param top_k: Nombre maximum de résultats à retourner.
        :return: Liste des contenus des documents les plus proches.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load or store embeddings first.")
        results = self.vector_store.similarity_search(query, top_k)
        return [result.page_content for result in results]


class VectorStoreManager:
    """
    Gère les opérations avancées sur la base vectorielle (initialisation, création de collections).

    :param db_path: Chemin de la base de données SQLite pour les vecteurs.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path

    def connect_to_db(self):
        """
        Initialise la connexion à la base de données SQLite. Crée une nouvelle base si elle n'existe pas.
        """
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            conn.close()

    def create_collection(self, collection_name: str):
        """
        Crée une nouvelle collection dans la base de données vectorielle.

        :param collection_name: Nom de la collection à créer.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {collection_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk TEXT,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()


def main():
    """
    Fonction principale pour tester les différentes fonctionnalités de la pipeline.
    """
    # Initialisation
    allowed_words = ["openai", "embedding", "pipeline", "pdf", "vector", "store"]
    manager = EnhancedLLMSecurityManager(allowed_words)
    pipeline = PDFPipeline(db_path="vectorstore_db")

    # Exemple d'utilisation
    prompt = "Can you process this PDF and store embeddings?"
    if manager.is_valid_prompt(prompt):
        sanitized_prompt = manager.sanitize_prompt(prompt)
        print(f"Sanitized Prompt: {sanitized_prompt}")

        # Chargement et traitement des PDF
        documents = pipeline.load_pdf("example.pdf")
        print(f"Loaded {len(documents)} pages from the PDF.")

        # Découpage et indexation
        split_docs = pipeline.split_text(documents)
        pipeline.store_embeddings(split_docs, collection_name="example_collection")
        print("Documents stored in vector database.")

        # Requêtes
        results = pipeline.query_embeddings("example query", top_k=3)
        print("Query Results:", results)
    else:
        print("Invalid prompt. Please use allowed terms only.")


if __name__ == "__main__":
    main()
