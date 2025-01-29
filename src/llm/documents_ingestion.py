import os
import re
import uuid
import fitz
import chromadb
import tiktoken
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

enc = tiktoken.get_encoding("o200k_base")

class BDDChunks:
    """
    A class to handle operations related to chunking text data, embedding, and storing in a ChromaDB instance.

    This class provides methods to:
    - Read text from PDF files.
    - Split the text into smaller chunks for processing.
    - Create a ChromaDB collection with embeddings for the chunks.
    - Add these chunks and their embeddings to the ChromaDB collection.
    """

    def __init__(self, embedding_model: str, path: str):
        """
        Initialize a BDDChunks instance.

        Args:
            embedding_model (str): The name of the embedding model to use for generating embeddings.
            path (str): The file path to the PDF or dataset to process.
        """
        self.path = path
        self.chunks: list[str] | None = None
        self.client = chromadb.PersistentClient(
            path="./ChromaDB", settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_name = embedding_model
        self.embeddings = SentenceTransformer(model_name=embedding_model)
        self.chroma_db = None

    def _create_collection(self, path: str) -> None:
        """
        Create a new ChromaDB collection for storing embeddings.

        Args:
            path (str): The name of the collection to create in ChromaDB.
        """
        # Tester qu'en changeant de path, on accède pas au reste
        file_name = "a" + os.path.basename(path)[0:50].strip() + "a"
        file_name = re.sub(r"\s+", "-", file_name)
        # Expected collection name that (1) contains 3-63 characters, (2) starts and ends with an alphanumeric character, (3) otherwise contains only alphanumeric characters, underscores or hyphens (-), (4) contains no two consecutive periods (..)
        self.chroma_db = self.client.get_or_create_collection(name=file_name, embedding_function=self.embeddings, metadata={"hnsw:space": "cosine"})  # type: ignore

    def read_pdf(self, file_path: str) -> str:
        """
        Reads the content of a PDF file, excluding the specified number of pages from the start and end.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text from the specified pages of the PDF.
        """
        doc = fitz.open(file_path)
        text = str()
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()  # type: ignore
        return text  # type: ignore

    def split_text_into_chunks(self, corpus: str, chunk_size: int = 500) -> list[str]:
        """
        Splits a given text corpus into chunks of a specified size.

        Args:
            corpus (str): The input text corpus to be split into chunks.
            chunk_size (int, optional): The size of each chunk. Defaults to 500.

        Returns:
            list[str]: A list of text chunks.
        """
        tokenized_corpus = enc.encode(corpus)
        chunks = [
            "".join(enc.decode(tokenized_corpus[i : i + chunk_size]))
            for i in tqdm(range(0, len(tokenized_corpus), chunk_size))
        ]

        return chunks

    def add_embeddings(self, list_chunks: list[str], batch_size: int = 100) -> None:
        """
        Add embeddings for text chunks to the ChromaDB collection.

        Args:
            list_chunks (list[str]): A list of text chunks to embed and add to the collection.
            batch_size (int, optional): The batch size for adding documents to the collection. Defaults to 100.

        Note:
            ChromaDB supports a maximum of 166 documents per batch.
        """
        if len(list_chunks) < batch_size:
            batch_size_for_chromadb = len(list_chunks)
        else:
            batch_size_for_chromadb = batch_size

        document_ids: list[str] = []

        for i in tqdm(
            range(0, len(list_chunks), batch_size_for_chromadb)
        ):  # On met en place une stratégie d'ajout par batch car ChromaDB ne supporte pas plus de 166 documents d'un coup.
            batch_documents = list_chunks[i : i + batch_size_for_chromadb]
            list_ids = [
                str(id_chunk) for id_chunk in list(range(i, i + len(batch_documents)))
            ]
            list_id_doc = [str(uuid.uuid4()) for x in list_ids]
            self.chroma_db.add(documents=batch_documents, ids=list_id_doc)  # type: ignore
            document_ids.extend(list_ids)

    def __call__(self) -> None:
        """
        Execute the entire process of reading, chunking, creating a collection, and adding embeddings.

        This method:
        1. Reads the text from the specified PDF file.
        2. Splits the text into chunks.
        3. Creates a ChromaDB collection for storing the embeddings.
        4. Adds the text chunks and their embeddings to the ChromaDB collection.
        """
        corpus = self.read_pdf(file_path=self.path)
        chunks = self.split_text_into_chunks(corpus=corpus)
        self._create_collection(path=self.path)
        self.add_embeddings(list_chunks=chunks)
