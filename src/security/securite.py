"""
Ce fichier contient un gestionnaire de sécurité pour le modèle de langage naturel (LLM).
"""

import re
import logging


class LLMSecurityManager:
    """
    Gestionnaire de sécurité pour le modèle de langage naturel (LLM).
    """

    def __init__(self, role : str = "educational assistant"):
        """
        Initialise le gestionnaire de sécurité pour le LLM.
        
        Args:
            role (str): Rôle de l'assistant (par défaut : "educational assistant").
        """
        self.system_prompt = (
            f"You are a {role}. "
            "Your purpose is to assist with educational content. "
            "Do not provide any information that is unrelated to this role."
        )
        self.forbidden_terms = ["hack", "bypass", "exploit", "malware", "confidential"]


    def clean_input(self, user_input : str) -> str:
        """
        Nettoie l'entrée utilisateur pour supprimer les caractères indésirables.
        
        Args:
            user_input (str): Input de l'utilisateur.

        Returns:
            str: Input nettoyé.
        """
        user_input = re.sub(r"[^\w\s,.?!]", "", user_input)
        return user_input[:200]


    def validate_input(self, user_input : str) -> tuple:
        """
        Valide si l'entrée utilisateur contient des termes interdits.
        
        Args:
            user_input (str): Input de l'utilisateur.

        Returns:
            tuple: Tuple (is_valid, message).
        """
        user_input_lower = user_input.lower()
        if any(term in user_input_lower for term in self.forbidden_terms):
            return False, "Requête bloquée pour des raisons de sécurité."
        return True, user_input


    def validate_output(self, output : str) -> tuple:
        """
        Valide si la sortie générée par le modèle contient des termes interdits.
        
        Args:
            output (str): Sortie générée par le modèle.

        Returns:
            tuple: Tuple (is_valid, message).
        """
        if any(term in output.lower() for term in self.forbidden_terms):
            return False, "Réponse bloquée pour des raisons de sécurité."
        return True, output


    def create_prompt(self, user_input : str) -> str:
        """
        Crée un prompt complet en ajoutant le contexte système.
        
        Args:
            user_input (str): Input de l'utilisateur.

        Returns:
            str: Prompt complet.
        """
        return f"{self.system_prompt}\n\nUser: {user_input}\nAssistant:"


    def log_interaction(self, user_input : str, response : str):
        """
        Enregistre les interactions entre l'utilisateur et le modèle dans un fichier de log.
        
        Args:
            user_input (str): Input de l'utilisateur.
            response (str): Réponse générée par le modèle.
        """
        logging.info("User Input: %s | Response: %s", user_input, response)


    def handle_blocked_request(self, reason : str) -> str:
        """
        Gère les requêtes bloquées en fournissant une réponse standardisée.
        
        Args:
            reason (str): Raison de blocage de la requête.

        Returns:
            str: Réponse standardisée pour les requêtes bloquées.
        """
        return (
            "Votre requête a été bloquée car elle enfreint nos règles. "
            f"Raison : {reason}. Veuillez poser une question éducative."
        )

# Fonction principale
if __name__ == "__main__":
    security_manager = LLMSecurityManager(role="educational assistant")

    # Nettoyage de l'input utilisateur
    USER_INPUT = "Explain how to hack a system! @#$%"
    cleaned_input = security_manager.clean_input(USER_INPUT)
    print("Input nettoyé :", cleaned_input)

    # Validation de l'input
    is_valid, message = security_manager.validate_input(cleaned_input)
    if not is_valid:
        print(security_manager.handle_blocked_request(message))
    else:
        # Création du prompt
        full_prompt = security_manager.create_prompt(cleaned_input)
        print("\nPrompt complet :\n", full_prompt)

        # Validation de la réponse
        RESPONSE = "This exploit can bypass systems."
        is_valid_response, validation_message = security_manager.validate_output(RESPONSE)
        if not is_valid_response:
            print("\nRéponse bloquée :", validation_message)
        else:
            print("\nRéponse validée :", RESPONSE)
