"""
Service AI unifié - Clients pour les 6 fournisseurs d'API
"""

import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from backend.config import settings, MODELS_CATALOG


@dataclass
class AIResponse:
    """Réponse standardisée d'un modèle AI"""

    content: str
    model_id: str
    model_name: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class BaseAIClient(ABC):
    """Client de base pour les APIs AI"""

    @abstractmethod
    @abstractmethod
    async def generate(
        self, model_id: str, prompt: str, history: List[Any] = None, **kwargs
    ) -> AIResponse:
        pass


class GroqClient(BaseAIClient):
    """Client pour l'API Groq"""

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    async def generate(
        self, model_id: str, prompt: str, history: List[Any] = None, **kwargs
    ) -> AIResponse:
        # Extraire le nom du modèle sans le préfixe provider
        model_name = model_id.replace("groq/", "")

        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        # Construire les messages avec l'historique
        messages = []
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

        # Ajouter le prompt actuel
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.BASE_URL, headers=headers, json=payload
                )
                response.raise_for_status()
                data = response.json()

                return AIResponse(
                    content=data["choices"][0]["message"]["content"],
                    model_id=model_id,
                    model_name=MODELS_CATALOG.get(model_id, {}).get("name", model_name),
                    provider="groq",
                    tokens_used=data.get("usage", {}).get("total_tokens"),
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
        except httpx.HTTPStatusError as e:
            return AIResponse(
                content="",
                model_id=model_id,
                model_name=model_name,
                provider="groq",
                error=f"Erreur HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return AIResponse(
                content="",
                model_id=model_id,
                model_name=model_name,
                provider="groq",
                error=str(e),
            )


class OpenRouterClient(BaseAIClient):
    """Client pour l'API OpenRouter"""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    async def generate(
        self, model_id: str, prompt: str, history: List[Any] = None, **kwargs
    ) -> AIResponse:
        model_name = model_id.replace("openrouter/", "")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Generative Site",
        }

        # Construire les messages avec l'historique
        messages = []
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.BASE_URL, headers=headers, json=payload
                )
                response.raise_for_status()
                data = response.json()

                return AIResponse(
                    content=data["choices"][0]["message"]["content"],
                    model_id=model_id,
                    model_name=MODELS_CATALOG.get(model_id, {}).get("name", model_name),
                    provider="openrouter",
                    tokens_used=data.get("usage", {}).get("total_tokens"),
                    finish_reason=data["choices"][0].get("finish_reason"),
                )
        except httpx.HTTPStatusError as e:
            return AIResponse(
                content="",
                model_id=model_id,
                model_name=model_name,
                provider="openrouter",
                error=f"Erreur HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return AIResponse(
                content="",
                model_id=model_id,
                model_name=model_name,
                provider="openrouter",
                error=str(e),
            )


class GeminiClient(BaseAIClient):
    """Client pour l'API Google Gemini (supporte le multimodal)"""

    async def generate(
        self, model_id: str, prompt: str, history: List[Any] = None, **kwargs
    ) -> AIResponse:
        model_name = model_id.replace("gemini/", "")
        images = kwargs.get("images", [])

        try:
            import google.generativeai as genai
            import base64

            genai.configure(api_key=settings.gemini_api_key)

            model = genai.GenerativeModel(model_name)

            # Gestion de l'historique pour Gemini
            chat_history = []
            if history:
                for msg in history:
                    # Mappage des rôles pour Gemini
                    role = "model" if msg.role == "assistant" else "user"
                    chat_history.append({"role": role, "parts": [msg.content]})

            # Si images présentes, on utilise generate_content directement (pas chat)
            # car le chat multimodal est complexe à gérer état par état
            if images:
                content_parts = [prompt]
                for img in images:
                    image_data = base64.b64decode(img["base64"])
                    content_parts.append({"mime_type": img["type"], "data": image_data})

                # Pour l'instant, si images, on ignore l'historique strict ou on le concatène
                # Gemini supporte l'historique multimodal mais c'est plus simple de concaténer le contexte textuel
                if history:
                    context_str = "\n".join(
                        [f"{msg.role}: {msg.content}" for msg in history]
                    )
                    content_parts.insert(0, f"Context:\n{context_str}\n\nTask:")

                response = model.generate_content(content_parts)
            else:
                # Mode chat standard
                chat = model.start_chat(history=chat_history)
                response = chat.send_message(prompt)

            return AIResponse(
                content=response.text,
                model_id=model_id,
                model_name=MODELS_CATALOG.get(model_id, {}).get("name", model_name),
                provider="gemini",
            )
        except Exception as e:
            return AIResponse(
                content="",
                model_id=model_id,
                model_name=model_name,
                provider="gemini",
                error=str(e),
            )


class HuggingFaceClient(BaseAIClient):
    """Client pour l'API HuggingFace Inference"""

    BASE_URL = "https://api-inference.huggingface.co/models"

    async def generate(
        self, model_id: str, prompt: str, history: List[Any] = None, **kwargs
    ) -> AIResponse:
        model_name = model_id.replace("huggingface/", "")

        headers = {
            "Authorization": f"Bearer {settings.huggingface_api_key}",
            "Content-Type": "application/json",
        }

        # Conversion historique en string pour modèles completion
        full_prompt = prompt
        if history:
            conversation = ""
            for msg in history:
                conversation += f"Role: {msg.role}\nContent: {msg.content}\n\n"
            full_prompt = f"{conversation}Role: user\nContent: {prompt}\n\nRole: assistant\nContent: "

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.7),
                "return_full_text": False,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/{model_name}", headers=headers, json=payload
                )
                response.raise_for_status()
                data = response.json()

                # HuggingFace retourne une liste
                content = (
                    data[0].get("generated_text", "")
                    if isinstance(data, list)
                    else data.get("generated_text", "")
                )

                return AIResponse(
                    content=content,
                    model_id=model_id,
                    model_name=MODELS_CATALOG.get(model_id, {}).get("name", model_name),
                    provider="huggingface",
                )
        except Exception as e:
            return AIResponse(
                content="",
                model_id=model_id,
                model_name=model_name,
                provider="huggingface",
                error=str(e),
            )


class CohereClient(BaseAIClient):
    """Client pour l'API Cohere"""

    async def generate(
        self, model_id: str, prompt: str, history: List[Any] = None, **kwargs
    ) -> AIResponse:
        model_name = model_id.replace("cohere/", "")

        try:
            import cohere

            co = cohere.Client(settings.cohere_api_key)

            # Format chat history for Cohere
            chat_history = []
            if history:
                for msg in history:
                    role = "CHATBOT" if msg.role == "assistant" else "USER"
                    chat_history.append({"role": role, "message": msg.content})

            response = co.chat(
                model=model_name,
                message=prompt,
                chat_history=chat_history,
                temperature=kwargs.get("temperature", 0.7),
            )

            return AIResponse(
                content=response.text,
                model_id=model_id,
                model_name=MODELS_CATALOG.get(model_id, {}).get("name", model_name),
                provider="cohere",
            )
        except Exception as e:
            return AIResponse(
                content="",
                model_id=model_id,
                model_name=model_name,
                provider="cohere",
                error=str(e),
            )


class CloudflareClient(BaseAIClient):
    """Client pour l'API Cloudflare Workers AI"""

    async def generate(
        self, model_id: str, prompt: str, history: List[Any] = None, **kwargs
    ) -> AIResponse:
        model_name = model_id.replace("cloudflare/", "")

        url = f"https://api.cloudflare.com/client/v4/accounts/{settings.cloudflare_account_id}/ai/run/{model_name}"

        headers = {
            "Authorization": f"Bearer {settings.cloudflare_api_key}",
            "Content-Type": "application/json",
        }

        # Construire les messages avec l'historique
        messages = []
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                content = data.get("result", {}).get("response", "")

                return AIResponse(
                    content=content,
                    model_id=model_id,
                    model_name=MODELS_CATALOG.get(model_id, {}).get("name", model_name),
                    provider="cloudflare",
                )
        except Exception as e:
            return AIResponse(
                content="",
                model_id=model_id,
                model_name=model_name,
                provider="cloudflare",
                error=str(e),
            )


class AIService:
    """
    Service unifié pour tous les fournisseurs AI.
    Route automatiquement les requêtes vers le bon client.
    """

    def __init__(self):
        self.clients: Dict[str, BaseAIClient] = {
            "groq": GroqClient(),
            "openrouter": OpenRouterClient(),
            "gemini": GeminiClient(),
            "huggingface": HuggingFaceClient(),
            "cohere": CohereClient(),
            "cloudflare": CloudflareClient(),
        }

    def _get_provider(self, model_id: str) -> str:
        """Extrait le provider depuis l'ID du modèle"""
        if model_id in MODELS_CATALOG:
            return MODELS_CATALOG[model_id]["provider"]
        # Fallback: extraire du préfixe
        return model_id.split("/")[0]

    async def generate(
        self, model_id: str, prompt: str, history: List[Any] = None, **kwargs
    ) -> AIResponse:
        """
        Génère une réponse avec le modèle spécifié.

        Args:
            model_id: Identifiant complet du modèle (ex: groq/llama-3.3-70b-versatile)
            prompt: Le prompt à envoyer
            history: Historique de la conversation (List[Message])
            **kwargs: Options additionnelles (temperature, max_tokens, etc.)

        Returns:
            AIResponse avec la réponse ou l'erreur
        """
        provider = self._get_provider(model_id)

        if provider not in self.clients:
            return AIResponse(
                content="",
                model_id=model_id,
                model_name="Unknown",
                provider=provider,
                error=f"Provider '{provider}' non supporté",
            )

        client = self.clients[provider]
        return await client.generate(model_id, prompt, history=history, **kwargs)


# Instance globale
ai_service = AIService()
