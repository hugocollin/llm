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
        max_tokens: int = 1000,
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
            Dict[str, Any]: Inclut la réponse et les métriques associées.
        """
        current_provider = provider or self.current_provider
        current_model = model or self.default_model

        if not (0.0 <= temperature <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        try:
            if current_provider == "mistral":
                response_text = await self._generate_mistral(prompt, current_model, temperature, max_tokens, **kwargs)
            elif current_provider == "gemini":
                response_text = self._generate_gemini(prompt, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Provider not supported: {current_provider}")
            
            # [TEMP] Exemple de métriques fictives : À REMPLACER !
            metrics = {
                "latency": 0,
                "euro_cost": 0,
                "energy_usage": 0,
                "gwp": 0,
            }
            
            return {
                "response": response_text,
                **metrics
            }
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return {"response": f"Error: {e}", "latency": 0.0, "euro_cost": 0.0, "energy_usage": 0.0, "gwp": 0.0}

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
            Dict[str, Any]: Configuration avec modèles, providers et capacités
        """
        providers = {
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
        }

        return {
            "current_provider": self.current_provider,
            "current_model": self.default_model,
            "providers": providers,
            "capabilities": ["text-generation", "multi-turn conversation"]
        }

    def switch_provider(self, provider: str, model: str):
        """
        Change le fournisseur et le modèle sélectionnés.

        Args:
            provider (str): Le fournisseur choisi.
            model (str): Le modèle choisi.

        Raises:
            ValueError: Si le fournisseur ou le modèle est invalide.
        """
        config = self.get_model_config()
        providers = config["providers"]

        if provider not in providers:
            raise ValueError(f"Fournisseur non supporté : {provider}")

        if model not in providers[provider]["models"]:
            raise ValueError(f"Modèle non supporté pour le fournisseur {provider} : {model}")

        self.current_provider = provider
        self.default_model = model