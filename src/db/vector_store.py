import os
import sys
import uuid
import fitz
import tiktoken
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm

class VectorStore:
    """
    Gère le stockage et la recherche de documents vectorisés avec ChromaDB.
    Fournit des fonctionnalités avancées de gestion de documents, incluant le traitement
    de PDFs et autres formats de documents.
    """

    def __init__(
        self,
        chemin_persistance: str = "../db/ChromaDB",
        modele_embedding: str = "mistral-embed",
        nom_collection: str = "collection_par_defaut",
        batch_size: int = 100
    ) -> None:
        """
        Initialise une instance de VectorStore.

        Args:
            chemin_persistance (str): Chemin pour persister les données ChromaDB.
            modele_embedding (str): Nom du modèle SentenceTransformer à utiliser.
            nom_collection (str): Nom de la collection par défaut.
            batch_size (int): Taille des lots pour l'ajout de documents.

        Raises:
            ValueError: Si les paramètres d'entrée sont invalides.
            RuntimeError: Si l'initialisation échoue.
        """
        if not isinstance(chemin_persistance, str) or not chemin_persistance.strip():
            raise ValueError("Le chemin de persistance doit être une chaîne non vide")
        if batch_size <= 0:
            raise ValueError("batch_size doit être supérieur à 0")

        self.batch_size = batch_size
        self.chemin_persistance = chemin_persistance
        os.makedirs(chemin_persistance, exist_ok=True)

        # Initialisation de l'encodeur tiktoken pour le découpage de texte
        try:
            self.encodeur = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            raise RuntimeError(f"Erreur d'initialisation de tiktoken: {str(e)}")

        try:
            self.client = chromadb.PersistentClient(
                path=chemin_persistance,
                settings=Settings(anonymized_telemetry=False)
            )
            self.fonction_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=modele_embedding
            )
            self.collection = self._creer_collection(nom_collection)
        except Exception as e:
            raise RuntimeError(f"Erreur d'initialisation de ChromaDB: {str(e)}")

    def _creer_collection(self, nom: str) -> chromadb.Collection:
        """
        Crée ou récupère une collection ChromaDB.

        Args:
            nom (str): Nom de la collection.

        Returns:
            chromadb.Collection: Collection ChromaDB créée ou existante.

        Raises:
            ValueError: Si le nom de la collection est invalide.
        """
        if not isinstance(nom, str) or not nom.strip():
            raise ValueError("Le nom de la collection doit être une chaîne non vide")

        return self.client.get_or_create_collection(
            name=nom,
            embedding_function=self.fonction_embedding,
            metadata={"hnsw:space": "cosine"}
        )

    def lire_pdf(self, chemin_fichier: str) -> str:
        """
        Lit le contenu d'un fichier PDF.

        Args:
            chemin_fichier (str): Chemin vers le fichier PDF.

        Returns:
            str: Le texte extrait du PDF.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si le fichier n'est pas un PDF valide.
        """
        if not os.path.exists(chemin_fichier):
            raise FileNotFoundError(f"Le fichier {chemin_fichier} n'existe pas")
        
        try:
            doc = fitz.open(chemin_fichier)
            texte = ""
            for page in doc:
                texte += page.get_text()
            return texte
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du PDF: {str(e)}")

    def decouper_texte(
        self,
        texte: str,
        taille_chunk: int = 500,
        chevauchement: int = 50
    ) -> List[str]:
        """
        Découpe un texte en chunks avec possibilité de chevauchement.

        Args:
            texte (str): Le texte à découper.
            taille_chunk (int): Taille de chaque chunk en tokens.
            chevauchement (int): Nombre de tokens de chevauchement entre chunks.

        Returns:
            List[str]: Liste des chunks de texte.

        Raises:
            ValueError: Si les paramètres sont invalides.
        """
        if not texte.strip():
            raise ValueError("Le texte ne peut pas être vide")
        if taille_chunk <= 0 or chevauchement < 0 or chevauchement >= taille_chunk:
            raise ValueError("Paramètres de taille de chunk invalides")

        tokens = self.encodeur.encode(texte)
        chunks = []
        
        for i in tqdm(range(0, len(tokens), taille_chunk - chevauchement)):
            chunk_tokens = tokens[i:i + taille_chunk]
            chunk_texte = self.encodeur.decode(chunk_tokens)
            chunks.append(chunk_texte)

        return chunks

    def ajouter_documents(
        self,
        textes: List[str],
        metadonnees: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Ajoute des documents à la collection par lots.

        Args:
            textes (List[str]): Liste des textes à ajouter.
            metadonnees (List[Dict], optional): Métadonnées associées aux textes.
            ids (List[str], optional): IDs personnalisés pour les documents.

        Returns:
            List[str]: Liste des IDs des documents ajoutés.

        Raises:
            ValueError: Si les paramètres sont invalides.
            RuntimeError: Si l'ajout des documents échoue.
        """
        if not textes:
            raise ValueError("La liste des textes ne peut pas être vide")

        if metadonnees and len(metadonnees) != len(textes):
            raise ValueError("Le nombre de métadonnées doit correspondre au nombre de textes")

        if ids and len(ids) != len(textes):
            raise ValueError("Le nombre d'IDs doit correspondre au nombre de textes")

        document_ids = ids or [str(uuid.uuid4()) for _ in textes]
        metadonnees = metadonnees or [{} for _ in textes]

        try:
            for i in tqdm(range(0, len(textes), self.batch_size)):
                batch_end = min(i + self.batch_size, len(textes))
                self.collection.add(
                    documents=textes[i:batch_end],
                    metadatas=metadonnees[i:batch_end],
                    ids=document_ids[i:batch_end]
                )
            return document_ids
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'ajout des documents: {str(e)}")

    def traiter_pdf(
        self,
        chemin_fichier: str,
        taille_chunk: int = 500,
        chevauchement: int = 50,
        metadonnees_supplementaires: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Traite un fichier PDF : lecture, découpage et stockage dans la base vectorielle.

        Args:
            chemin_fichier (str): Chemin vers le fichier PDF.
            taille_chunk (int): Taille des chunks en tokens.
            chevauchement (int): Chevauchement entre chunks.
            metadonnees_supplementaires (Dict, optional): Métadonnées additionnelles.

        Returns:
            List[str]: Liste des IDs des chunks stockés.

        Raises:
            Exception: Si une erreur survient pendant le traitement.
        """
        try:
            # Lecture du PDF
            texte = self.lire_pdf(chemin_fichier)
            
            # Découpage en chunks
            chunks = self.decouper_texte(texte, taille_chunk, chevauchement)
            
            # Préparation des métadonnées
            metadonnees_base = {
                "source": chemin_fichier,
                "type": "pdf",
                "taille_chunk": taille_chunk,
                "chevauchement": chevauchement
            }
            if metadonnees_supplementaires:
                metadonnees_base.update(metadonnees_supplementaires)
            
            metadonnees = [
                {**metadonnees_base, "chunk_index": i, "total_chunks": len(chunks)}
                for i in range(len(chunks))
            ]
            
            # Stockage dans la base vectorielle
            return self.ajouter_documents(chunks, metadonnees)
            
        except Exception as e:
            raise Exception(f"Erreur lors du traitement du PDF: {str(e)}")

    def traiter_dossier_pdf(
        self,
        chemin_dossier: str,
        taille_chunk: int = 500,
        chevauchement: int = 50,
        metadonnees_supplementaires: Optional[Dict[str, Any]] = None,
        recursif: bool = False
    ) -> Dict[str, List[str]]:
        """
        Traite tous les fichiers PDF d'un dossier.

        Args:
            chemin_dossier (str): Chemin vers le dossier contenant les PDFs.
            taille_chunk (int): Taille des chunks en tokens.
            chevauchement (int): Chevauchement entre chunks.
            metadonnees_supplementaires (Dict, optional): Métadonnées additionnelles.
            recursif (bool): Traiter les sous-dossiers récursivement.

        Returns:
            Dict[str, List[str]]: Dictionnaire {nom_fichier: liste_ids_chunks}.

        Raises:
            Exception: Si une erreur survient pendant le traitement.
        """
        if not os.path.exists(chemin_dossier):
            raise FileNotFoundError(f"Le dossier {chemin_dossier} n'existe pas")

        resultats = {}
        pattern = "**/*.pdf" if recursif else "*.pdf"

        try:
            for chemin_pdf in tqdm(list(Path(chemin_dossier).glob(pattern))):
                try:
                    ids = self.traiter_pdf(
                        str(chemin_pdf),
                        taille_chunk,
                        chevauchement,
                        metadonnees_supplementaires
                    )
                    resultats[str(chemin_pdf)] = ids
                except Exception as e:
                    print(f"Erreur lors du traitement de {chemin_pdf}: {str(e)}")
                    continue

            return resultats
        except Exception as e:
            raise Exception(f"Erreur lors du traitement du dossier: {str(e)}")

    def rechercher(
        self,
        requete: Union[str, List[str]],
        nb_resultats: int = 3,
        filtre_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Recherche les documents les plus similaires à la requête.

        Args:
            requete (Union[str, List[str]]): Texte ou liste de textes de requête.
            nb_resultats (int): Nombre de résultats à retourner.
            filtre_dict (Dict, optional): Filtres à appliquer sur les métadonnées.
            include_metadata (bool): Inclure les métadonnées dans les résultats.

        Returns:
            Dict[str, Any]: Résultats de la recherche contenant :
                - 'documents': Liste des documents trouvés
                - 'metadonnees': Liste des métadonnées associées
                - 'distances': Liste des scores de similarité
                - 'ids': Liste des IDs des documents

        Raises:
            ValueError: Si les paramètres sont invalides.
            RuntimeError: Si la recherche échoue.
        """
        if isinstance(requete, str):
            requete = [requete]
        if not requete or not any(requete):
            raise ValueError("La requête ne peut pas être vide")
        if nb_resultats <= 0:
            raise ValueError("nb_resultats doit être supérieur à 0")

        try:
            resultats = self.collection.query(
                query_texts=requete,
                n_results=nb_resultats,
                where=filtre_dict,
                include=["documents", "metadatas", "distances", "embeddings"] if include_metadata else ["documents", "distances"]
            )

            return {
                "documents": resultats["documents"],
                "metadonnees": resultats.get("metadatas", []),
                "distances": resultats["distances"],
                "ids": resultats.get("ids", [])
            }
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la recherche: {str(e)}")

    def obtenir_statistiques_collection(self) -> Dict[str, Any]:
        """
        Récupère les statistiques détaillées de la collection.

        Returns:
            Dict[str, Any]: Statistiques de la collection contenant :
                - 'nombre': Nombre de documents
                - 'nom': Nom de la collection
                - 'metadata': Métadonnées de la collection
                - 'dimension': Dimension des embeddings

        Raises:
            RuntimeError: Si la récupération des statistiques échoue.
        """
        try:
            return {
                "nombre": self.collection.count(),
                "nom": self.collection.name,
                "metadata": self.collection.metadata,
                "dimension": len(next(iter(self.collection.get(limit=1)["embeddings"]))) if self.collection.count() > 0 else None
            }
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la récupération des statistiques: {str(e)}")

    def supprimer_documents(self, ids: List[str]) -> None:
        """
        Supprime des documents de la collection.

        Args:
            ids (List[str]): Liste des IDs des documents à supprimer.

        Raises:
            ValueError: Si la liste des IDs est vide.
            RuntimeError: Si la suppression échoue.
        """
        if not ids:
            raise ValueError("La liste des IDs ne peut pas être vide")

        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la suppression des documents: {str(e)}")

    def modifier_document(
        self,
        id: str,
        nouveau_texte: Optional[str] = None,
        nouvelles_metadonnees: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Modifie un document existant dans la collection.

        Args:
            id (str): ID du document à modifier.
            nouveau_texte (str, optional): Nouveau texte du document.
            nouvelles_metadonnees (Dict, optional): Nouvelles métadonnées.

        Raises:
            ValueError: Si aucune modification n'est spécifiée.
            RuntimeError: Si la modification échoue.
        """
        if nouveau_texte is None and nouvelles_metadonnees is None:
            raise ValueError("Au moins un paramètre de modification doit être spécifié")

        try:
            update_params = {}
            if nouveau_texte is not None:
                update_params["documents"] = [nouveau_texte]
            if nouvelles_metadonnees is not None:
                update_params["metadatas"] = [nouvelles_metadonnees]

            self.collection.update(ids=[id], **update_params)
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la modification du document: {str(e)}")