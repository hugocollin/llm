from securite import LLMSecurityManager
from promptClassifier import PromptClassifier

class EnhancedLLMSecurityManager(LLMSecurityManager):
    def __init__(self, role="educational assistant", train_json_path=None, test_json_path=None):
        """
        Initialise le gestionnaire de sécurité amélioré avec un classifieur.
        
        :param role: Rôle du modèle.
        :param train_json_path: Chemin vers le fichier JSON d'entraînement.
        :param test_json_path: Chemin vers le fichier JSON de test.
        """
        super().__init__(role)
        self.classifier = PromptClassifier()

        # Charger et entraîner le classifieur si les chemins sont fournis
        if train_json_path and test_json_path:
            self.classifier.load_train_and_test_data_from_json(train_json_path, test_json_path)
            self.classifier.train_and_evaluate()
            self.classifier.get_best_model()
            self.classifier.export_best_model("best_prompt_model.pkl")

    def validate_input(self, user_input):
        """
        Valide une entrée utilisateur en utilisant des règles et le classifieur ML.
        
        :param user_input: Input de l'utilisateur.
        :return: Tuple (is_valid, message).
        """
        # Nettoyer l'entrée
        cleaned_input = self.clean_input(user_input)

        # Valider avec les règles
        is_valid, message = super().validate_input(cleaned_input)
        if not is_valid:
            return is_valid, message


        # Valider avec le classifieur ML
        prediction = self.classifier.predict_with_best_model([cleaned_input])[0]
        if prediction == 1:  # 1 = malveillant
            return False, "Requête bloquée : intention malveillante détectée par le modèle ML."

        return True, "Requête valide."

# Exemple d'utilisation
if __name__ == "__main__":
    # Fichiers JSON d'entraînement et de test
    train_json = "guardrail_dataset_train.json"
    test_json = "guardrail_dataset_test.json"

    # Initialisation du gestionnaire
    enhanced_security_manager = EnhancedLLMSecurityManager(
        role="educational assistant",
        train_json_path=train_json,
        test_json_path=test_json
    )

    # Exemple de prompt utilisateur
    user_input = "Cancer provision women Germany"
    is_valid, message = enhanced_security_manager.validate_input(user_input)
    print(message)
