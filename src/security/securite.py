import re
import logging

class LLMSecurityManager:
    def __init__(self, role="educational assistant"):
        """
        Initialise le gestionnaire de sécurité pour le LLM.
        
        :param role: Rôle du modèle (par défaut : 'educational assistant').
        """
        self.system_prompt = (
            f"You are a {role}. "
            "Your purpose is to assist with educational content. "
            "Do not provide any information that is unrelated to this role."
        )
        self.forbidden_terms = ["hack", "bypass", "exploit", "malware", "confidential"]
        
        # # Configuration du journal
        # logging.basicConfig(
        #     filename="user_interactions.log",
        #     level=logging.INFO,
        #     format="%(asctime)s - %(message)s"
        # )

    def clean_input(self, user_input):
        """
        Nettoie l'entrée utilisateur pour supprimer les caractères indésirables.
        
        :param user_input: Input de l'utilisateur.
        :return: Input nettoyé.
        """
        user_input = re.sub(r"[^\w\s,.?!]", "", user_input)
        return user_input[:200]

    def validate_input(self, user_input):
        """
        Valide si l'entrée utilisateur contient des termes interdits.
        
        :param user_input: Input de l'utilisateur.
        :return: Tuple (is_valid, message).
        """
        user_input_lower = user_input.lower()
        if any(term in user_input_lower for term in self.forbidden_terms):
            return False, "Requête bloquée pour des raisons de sécurité."
        return True, user_input

    def validate_output(self, output):
        """
        Valide si la sortie générée par le modèle contient des termes interdits.
        
        :param output: Réponse générée par le LLM.
        :return: Tuple (is_valid, message).
        """
        if any(term in output.lower() for term in self.forbidden_terms):
            return False, "Réponse bloquée pour des raisons de sécurité."
        return True, output

    def create_prompt(self, user_input):
        """
        Crée un prompt complet en ajoutant le contexte système.
        
        :param user_input: Input de l'utilisateur.
        :return: Prompt complet.
        """
        return f"{self.system_prompt}\n\nUser: {user_input}\nAssistant:"

    def log_interaction(self, user_input, response):
        """
        Enregistre les interactions entre l'utilisateur et le modèle dans un fichier de log.
        
        :param user_input: Input de l'utilisateur.
        :param response: Réponse générée par le modèle.
        """
        logging.info(f"User Input: {user_input} | Response: {response}")

    def handle_blocked_request(self, reason):
        """
        Gère les requêtes bloquées en fournissant une réponse standardisée.
        
        :param reason: Raison du blocage.
        :return: Message pour l'utilisateur.
        """
        return f"Votre requête a été bloquée car elle enfreint nos règles. Raison : {reason}. Veuillez poser une question éducative."


# Exemple d'utilisation
if __name__ == "__main__":
    security_manager = LLMSecurityManager(role="educational assistant")

    # Nettoyage de l'input utilisateur
    user_input = "Explain how to hack a system! @#$%"
    cleaned_input = security_manager.clean_input(user_input)
    print("Input nettoyé :", cleaned_input)

    # Validation de l'input
    is_valid, message = security_manager.validate_input(cleaned_input)
    if not is_valid:
        print(security_manager.handle_blocked_request(message))
    else:
        # Création du prompt
        full_prompt = security_manager.create_prompt(cleaned_input)
        print("\nPrompt complet :\n", full_prompt)

        # Exemple de validation de la réponse
        response = "This exploit can bypass systems."
        is_valid_response, validation_message = security_manager.validate_output(response)
        if not is_valid_response:
            print("\nRéponse bloquée :", validation_message)
        else:
            print("\nRéponse validée :", response)

        # Journalisation de l'interaction
        security_manager.log_interaction(cleaned_input, response)



