import torch
from transformers import BertTokenizer, BertModel
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import json


class PromptClassifier:
    def __init__(self, bert_model_name: str = 'bert-base-multilingual-uncased') -> None:
        """
        Initialise la classe avec le modèle BERT et le tokenizer.

        Args:
            bert_model_name (str): Nom du modèle pré-entraîné BERT à utiliser.
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.estimators = {
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": svm.SVC(),
            "Random Forest": RandomForestClassifier()
        }
        self.results = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1 score"])
        self.best_model = None

    def get_bert_embedding(self, prompt: str) -> np.ndarray:
        """
        Tokenise le texte et retourne l'embedding généré par BERT.

        Args:
            prompt (str): Texte du prompt à encoder.

        Returns:
            np.ndarray: Vecteur d'embedding généré par BERT.
        """
        tokens = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**tokens)
        last_hidden_states = outputs.last_hidden_state
        embedding_vector = last_hidden_states.mean(dim=1).squeeze().numpy()
        return embedding_vector

    def train_and_evaluate(self) -> pd.DataFrame:
        """
        Entraîne et évalue chaque modèle avec les données préparées.

        Returns:
            pd.DataFrame: Résultats des métriques de performance des modèles.
        """
        for est_name, est_obj in self.estimators.items():
            # Entraîner le modèle
            est_obj.fit(self.X_train_emb, self.y_train)

            # Prédire sur les données de test
            y_predict = est_obj.predict(self.X_test_emb)

            # Calculer les métriques de performance
            accuracy = accuracy_score(self.y_test, y_predict)
            precision = precision_score(self.y_test, y_predict)
            recall = recall_score(self.y_test, y_predict)
            f1 = f1_score(self.y_test, y_predict)

            # Stocker les résultats
            self.results.loc[est_name] = [accuracy, precision, recall, f1]

        return self.results

    def get_best_model(self) -> Tuple[str, Dict[str, float]]:
        """
        Identifie le meilleur modèle basé sur le score F1.

        Returns:
            Tuple[str, Dict[str, float]]: Nom du meilleur modèle et ses métriques.
        """
        if self.results.empty:
            raise ValueError("Les modèles n'ont pas encore été entraînés. Appelez `train_and_evaluate` d'abord.")

        # Identifier le modèle avec le meilleur score F1
        best_model_name = self.results["f1 score"].idxmax()
        best_metrics = self.results.loc[best_model_name].to_dict()

        # Sauvegarder le meilleur modèle
        self.best_model = (best_model_name, self.estimators[best_model_name])

        return best_model_name, best_metrics

    def predict_with_best_model(self, prompts: List[str]) -> List[int]:
        """
        Utilise le meilleur modèle pour prédire une liste de prompts.

        Args:
            prompts (List[str]): Liste des prompts à analyser.

        Returns:
            List[int]: Liste des labels prédits (0 ou 1).
        """
        if self.best_model is None:
            raise ValueError("Aucun modèle sélectionné. Appelez `get_best_model` d'abord.")

        _, model = self.best_model

        # Générer les embeddings pour les nouveaux prompts
        embeddings = [self.get_bert_embedding(prompt) for prompt in prompts]
        embeddings = pd.DataFrame(embeddings)

        # Faire les prédictions
        predictions = model.predict(embeddings)
        return predictions

    def load_train_and_test_data_from_json(self, train_json_path: str, test_json_path: str) -> None:
        """
        Charge les données d'entraînement et de test à partir de deux fichiers JSON distincts.

        Args:
            train_json_path (str): Chemin vers le fichier JSON contenant les données d'entraînement.
            test_json_path (str): Chemin vers le fichier JSON contenant les données de test.

        Returns:
            None
        """
        # Charger les données d'entraînement
        with open(train_json_path, 'r', encoding='utf-8') as f_train:
            train_data = json.load(f_train)

        # Charger les données de test
        with open(test_json_path, 'r', encoding='utf-8') as f_test:
            test_data = json.load(f_test)

        # Vérifier que les clés "prompt" et "label" existent dans les deux fichiers
        for data, dataset_name in zip([train_data, test_data], ["train", "test"]):
            if not all(key in data[0] for key in ['prompt', 'label']):
                raise ValueError(f"Le fichier JSON pour {dataset_name} doit contenir les clés 'prompt' et 'label'.")

        # Extraire les prompts et labels des ensembles d'entraînement et de test
        X_train = [item['prompt'] for item in train_data]
        y_train = [item['label'] for item in train_data]
        X_test = [item['prompt'] for item in test_data]
        y_test = [item['label'] for item in test_data]

        # Préparer les données
        self.prepare_data(X_train, y_train, X_test, y_test)