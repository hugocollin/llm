import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio

import google.generativeai as genai
from mistralai import Mistral


class LLMBase(ABC):
    """Classe de base abstraite pour les modèles de langage."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse à partir d'un prompt.

        Args:
            prompt (str): Texte d'entrée pour la génération
            **kwargs: Arguments supplémentaires

        Returns:
            str: Réponse générée
        """
        pass

    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """
        Récupère la configuration du modèle.

        Returns:
            Dict[str, Any]: Configuration détaillée du modèle
        """
        pass


class MultiModelLLM(LLMBase):
    def __init__(
        self,
        api_key_mistral: Optional[str] = None,
        api_key_gemini: Optional[str] = None,
        default_model: str = "large",
        default_provider: str = "mistral",
    ):
        """
        Initialise le gestionnaire multi-modèles.

        Args:
            api_key_mistral: Clé API pour Mistral.
            api_key_gemini: Clé API pour Gemini.
            default_model: Modèle par défaut.
            default_provider: Fournisseur par défaut.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.api_key_mistral = api_key_mistral or os.environ.get("MISTRAL_API_KEY")
        self.api_key_gemini = api_key_gemini or os.environ.get("GEMINI_API_KEY")

        if not self.api_key_mistral:
            raise ValueError("MISTRAL_API_KEY is missing")
        if not self.api_key_gemini:
            raise ValueError("GEMINI_API_KEY is missing")

        self.mistral_client = Mistral(api_key=self.api_key_mistral)
        genai.configure(api_key=self.api_key_gemini)

        self.default_model = default_model
        self.current_provider = default_provider

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        **kwargs,
    ) -> str:
        """
        Génère une réponse.

        Args:
            prompt: Le prompt.
            model: Le modèle à utiliser.
            provider: Le fournisseur (mistral ou gemini).
            temperature: La température.
            max_tokens: Le nombre maximum de tokens.
            **kwargs: Arguments additionnels spécifiques au fournisseur.

        Returns:
            La réponse générée.
        """
        current_provider = provider or self.current_provider
        current_model = model or self.default_model

        if not (0.0 <= temperature <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        try:
            if current_provider == "mistral":
               return asyncio.run(self._generate_mistral(prompt, current_model, temperature, max_tokens, **kwargs))
            elif current_provider == "gemini":
                return self._generate_gemini(prompt, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Provider not supported: {current_provider}")
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return f"Error: {e}"

    async def _generate_mistral(self, prompt: str, model: str, temperature: float, max_tokens: int, **kwargs) -> str:
        """Génère une réponse avec Mistral.

        Args:
            prompt: Le prompt.
            model: Le modèle Mistral (ex: Mistral-Large-Instruct-2411).
            temperature: La température.
            max_tokens: Le nombre maximum de tokens.
            **kwargs: Arguments additionnels pour l'API Mistral (ex: top_p).

        Returns:
            La réponse générée.
        """
        try:
            chat_response = await self.mistral_client.chat.complete_async(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            if chat_response.choices:
                return chat_response.choices[0].message.content
            else:
                self.logger.warning("No choices returned by Mistral API.")
                return ""
        except Exception as e:
            self.logger.error(f"Mistral API error: {e}")
            return f"Mistral API Error: {e}"


    def _generate_gemini(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
        """
        Génère une réponse avec Gemini.

        Args:
            prompt: Le prompt.
            temperature: La température.
            max_tokens: Le nombre maximum de tokens.
            **kwargs: Arguments additionnels pour l'API Gemini.

        Returns:
            La réponse générée.
        """
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs,
                ),
            )
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return f"Gemini API Error: {e}"

    def get_model_config(self) -> Dict[str, Any]:
        """
        Récupère la configuration détaillée du modèle.

        Returns:
            Dict[str, Any]: Configuration avec modèles et capacités
        """
        return {
            "current_provider": self.current_provider,
            "current_model": self.default_model,
            "providers": {
                "mistral": {
                    "models": [
                        "Mistral-Large-Instruct-2411",
                        "Mistral-Small-Instruct-2409"
                    ]
                },
                "gemini": {
                    "models": [
                        "gemini-1.5-flash",
                        "gemini-1.5-pro"
                    ]
                }
            },
            "capabilities": ["text-generation", "multi-turn conversation"]
        }


    def switch_provider(self, new_provider: str, new_model_name: Optional[str] = None):
        """
        Dynamically switch model provider with optional model name update.

        Args:
            new_provider (str): Target provider.
            new_model_name (str, optional): Specific model variant.
        """
        supported_providers = ["mistral", "gemini"]
        if new_provider not in supported_providers:
            raise ValueError(f"Provider must be one of {supported_providers}")

        self.logger.info(f"Switching provider to {new_provider} with model {new_model_name or self.default_model}")
        self.current_provider = new_provider
        if new_model_name:
            self.default_model = new_model_name