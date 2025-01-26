import chromadb
from typing import List, Dict, Any
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class VectorStore:
    """
    Gère le stockage et la recherche de documents vectorisés avec ChromaDB.
    Utilise SentenceTransformer pour générer les embeddings.
    """

    def __init__(self, chemin_persistance: str = "./ChromaDB", modele_embedding: str = "all-MiniLM-L6-v2"):
        """
        Initialise une instance de MagasinDeVecteurs.

        Args:
            chemin_persistance (str): Chemin pour persister les données ChromaDB.
            modele_embedding (str): Nom du modèle SentenceTransformer à utiliser.

        Returns:
            None
        """
        self.client = chromadb.PersistentClient(
            path=chemin_persistance,
            settings=Settings(anonymized_telemetry=False)
        )
        self.fonction_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=modele_embedding)
        self.collection = self._creer_collection()
    
    def _creer_collection(self, nom: str = "collection_par_defaut") -> chromadb.Collection:
        """
        Crée ou récupère une collection ChromaDB.

        Args:
            nom (str): Nom de la collection.

        Returns:
            chromadb.Collection: Collection ChromaDB créée ou existante.
        """
        return self.client.get_or_create_collection(
            name=nom,
            embedding_function=self.fonction_embedding,
            metadata={"hnsw:space": "cosine"}
        )
    
    def ajouter_documents(self, textes: List[str], metadonnees: List[Dict[str, Any]] = None) -> None:
        """
        Ajoute des documents à la collection.

        Args:
            textes (List[str]): Liste des textes à ajouter.
            metadonnees (List[Dict]): Métadonnées associées aux textes.

        Returns:
            None
        """
        if not metadonnees:
            metadonnees = [{} for _ in textes]
            
        self.collection.add(
            documents=textes,
            metadatas=metadonnees,
            ids=[f"doc_{i}" for i in range(len(textes))]
        )
    
    def rechercher(self, requete: str, nb_resultats: int = 3, filtre_dict: Dict[str, Any] = None) -> Dict[str, List]:
        """
        Recherche les documents les plus similaires à la requête.

        Args:
            requete (str): Texte de la requête.
            nb_resultats (int): Nombre de résultats à retourner.
            filtre_dict (Dict): Filtres à appliquer sur les métadonnées.

        Returns:
            Dict[str, List]: Dictionnaire contenant :
                - 'documents': Liste des documents trouvés.
                - 'metadonnees': Liste des métadonnées associées.
                - 'distances': Liste des scores de similarité.
        """
        resultats = self.collection.query(
            query_texts=[requete],
            n_results=nb_resultats,
            where=filtre_dict
        )
        return {
            "documents": resultats["documents"],
            "metadonnees": resultats["metadatas"],
            "distances": resultats["distances"]
        }
    
    def obtenir_statistiques_collection(self) -> Dict[str, Any]:
        """
        Récupère les statistiques de la collection.

        Returns:
            Dict[str, Any]: Dictionnaire contenant :
                - 'nombre': Nombre de documents.
                - 'nom': Nom de la collection.
        """
        return {
            "nombre": self.collection.count(),
            "nom": self.collection.name
        }
