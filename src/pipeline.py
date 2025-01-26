from src.security.securite import LLMSecurityManager
from src.ml.promptClassifier import PromptClassifier
import os


class EnhancedLLMSecurityManager(LLMSecurityManager):
    def __init__(self, user_input, role="educational assistant", train_json_path=None, test_json_path=None, train_model=False):
        """
        Initialise le gestionnaire de sécurité amélioré avec un classifieur et un prompt utilisateur.

        :param user_input: Entrée utilisateur à valider.
        :param role: Rôle du modèle.
        :param train_json_path: Chemin vers le fichier JSON d'entraînement.
        :param test_json_path: Chemin vers le fichier JSON de test.
        :param train_model: Booléen indiquant si le modèle doit être entraîné.
        """
        super().__init__(role)
        self.user_input = user_input  # Enregistrer l'input utilisateur
        self.classifier = PromptClassifier()

        # Charger et entraîner le classifieur si demandé
        if train_model and train_json_path and test_json_path:
            print("Training the model...")
            self.classifier.load_train_and_test_data_from_json(train_json_path, test_json_path)
            self.classifier.train_and_evaluate()
            self.classifier.get_best_model()
            self.classifier.export_best_model("best_prompt_model.pkl")
        elif not train_model:
            print("Loading the model...")
            self.classifier.load_model()

    def validate_input(self):
        """
        Valide une entrée utilisateur en utilisant des règles et le classifieur ML.

        :return: Tuple (is_valid, message).
        """
        # Nettoyer l'entrée
        cleaned_input = self.clean_input(self.user_input)

        # Valider avec les règles
        is_valid, _ = super().validate_input(cleaned_input)
        if not is_valid:
            return is_valid

        # Valider avec le classifieur ML
        prediction = self.classifier.predict_with_best_model([cleaned_input])[0]
        if prediction == 1:  # 1 = malveillant
            return False

        return True


# Exemple d'utilisation
if __name__ == "__main__":
    # Fichiers JSON d'entraînement et de test
    train_json = "guardrail_dataset_train.json"
    test_json = "guardrail_dataset_test.json"
    train_json_path = os.path.join("src", "ml", train_json)
    test_json_path = os.path.join("src", "ml", test_json)

    # Prompt utilisateur
    user_input = "SELECT * FROM users WHERE id=1;"

    # Initialisation du gestionnaire
    enhanced_security_manager = EnhancedLLMSecurityManager(
        user_input=user_input,
        role="educational assistant",
        train_json_path=train_json_path,
        test_json_path=test_json_path,
        train_model=False # Mettre à True pour ré-entraîner le modèle
    )

    # Validation de l'entrée utilisateur
    is_valid = enhanced_security_manager.validate_input()
    print(is_valid)
