import os
import logging
import time
import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from mistralai import Mistral

# Decorator to measure latency
def measure_latency(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        self.last_latency = end_time - start_time
        return result
    return wrapper

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
        default_model: str = "ministral-8b-latest",
        default_provider: str = "mistral",
        default_temperature: float = 0.7,
    ):
        """
        Initialise le gestionnaire multi-modèles.

        Args:
            api_key_mistral: Clé API pour Mistral.
            api_key_gemini: Clé API pour Gemini.
            default_model: Modèle par défaut.
            default_provider: Fournisseur par défaut.
            default_temperature: Température par défaut pour la génération.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.api_key_mistral = api_key_mistral or os.environ.get("MISTRAL_API_KEY")
        self.api_key_gemini = api_key_gemini or os.environ.get("GEMINI_API_KEY")

        if not self.api_key_mistral:
            self.logger.error("MISTRAL_API_KEY is missing. Please set it in the environment or pass it explicitly.")
            raise ValueError("MISTRAL_API_KEY is missing")
        if not self.api_key_gemini:
            self.logger.error("GEMINI_API_KEY is missing. Please set it in the environment or pass it explicitly.")
            raise ValueError("GEMINI_API_KEY is missing")

        self.mistral_client = Mistral(api_key=self.api_key_mistral)
        genai.configure(api_key=self.api_key_gemini)

        self.default_model = default_model
        self.default_provider = default_provider
        self.current_provider = default_provider
        self.default_temperature = default_temperature
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.last_latency = 0.0

        # Validate default provider and model
        if self.default_provider not in self.get_model_config()["providers"]:
            raise ValueError(f"Default provider '{self.default_provider}' is not supported.")
        if self.default_model not in self.get_model_config()["providers"][self.default_provider]["models"]:
            raise ValueError(f"Default model '{self.default_model}' is not supported for provider '{self.default_provider}'.")

    @measure_latency
    def _generate_mistral(self, prompt, model, temperature, max_tokens, **kwargs):
        try:
            chat_response = self.mistral_client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            energy_usage, gwp = self._get_energy_usage(chat_response)
            return chat_response.choices[0].message.content, energy_usage, gwp
        except Exception as e:
            self.logger.error(f"Mistral API error: {e}")
            return f"Mistral API Error: {e}", 0.0, 0.0

    @measure_latency
    def _generate_gemini(self, prompt, temperature, max_tokens, **kwargs):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature, max_output_tokens=max_tokens, **kwargs
                )
            )
            energy_usage, gwp = self._get_energy_usage(response)
            return response.text, energy_usage, gwp
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return f"Gemini API Error: {e}", 0.0, 0.0

    def _get_energy_usage(self, response: Any) -> Tuple[float, float]:
        energy_usage = getattr(response, "impacts", {}).get("energy", {}).get("value", 0.0)
        gwp = getattr(response, "impacts", {}).get("gwp", {}).get("value", 0.0)
        return energy_usage, gwp

    def _get_price_query(self, llm_name: str, input_tokens: int, output_tokens: int) -> float:
        pricing = {
            "ministral-8b-latest": {"input": 0.095, "output": 0.095},
            "mistral-large-latest": {"input": 1.92, "output": 5.75},
            "codestral-latest": {"input": 0.30, "output": 0.85},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
            "gemini-1.5-pro": {"input": 1.20, "output": 4.80}
        }
        if llm_name not in pricing:
            raise ValueError(f"LLM {llm_name} not found in pricing database.")
        cost_input = (input_tokens / 1_000_000) * pricing[llm_name]["input"]
        cost_output = (output_tokens / 1_000_000) * pricing[llm_name]["output"]
        return cost_input + cost_output

    def generate(self, prompt: str, model: Optional[str] = None, provider: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        current_provider = provider or self.current_provider
        current_model = model or self.default_model

        try:
            if current_provider == "mistral":
                response_text, energy_usage, gwp = self._generate_mistral(prompt, current_model, temperature, max_tokens, **kwargs)
            elif current_provider == "gemini":
                response_text, energy_usage, gwp = self._generate_gemini(prompt, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Provider not supported: {current_provider}")

            cost = self._get_price_query(current_model, input_tokens=100, output_tokens=500)  # Replace with actual token counts

            return {
                "response": response_text,
                "latency": self.last_latency,
                "euro_cost": cost,
                "energy_usage": energy_usage,
                "gwp": gwp,
            }
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return {"response": f"Error: {e}", "latency": 0.0, "euro_cost": 0.0, "energy_usage": 0.0, "gwp": 0.0}

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "providers": {
                "mistral": {
                    "models": ["ministral-8b-latest", "mistral-large-latest", "codestral-latest"]
                },
                "gemini": {
                    "models": ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
                }
            }
        }

    def switch_provider(self, provider: str, model: str, temperature: float):
        config = self.get_model_config()
        providers = config["providers"]

        if provider not in providers:
            raise ValueError(f"Fournisseur non supporté : {provider}")

        if model not in providers[provider]["models"]:
            raise ValueError(f"Modèle non supporté pour le fournisseur {provider} : {model}")

        if not (0.0 <= temperature <= 1.0):
            raise ValueError("La température doit être comprise entre 0.0 et 1.0")

        self.current_provider = provider
        self.default_model = model
        self.default_temperature = temperature