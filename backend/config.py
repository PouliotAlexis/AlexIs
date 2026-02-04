"""
Configuration centralisée pour le site AI Générative Multi-Agents
"""

from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
import os


class Settings(BaseSettings):
    """Configuration depuis les variables d'environnement"""

    # API Keys
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    gemini_api_key: str = ""
    huggingface_api_key: str = ""
    cohere_api_key: str = ""
    cloudflare_account_id: str = ""
    cloudflare_api_key: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Catalogue des modèles disponibles avec leurs caractéristiques
MODELS_CATALOG: Dict[str, dict] = {
    # === GROQ ===
    "groq/llama-3.3-70b-versatile": {
        "provider": "groq",
        "name": "Llama 3.3 70B",
        "strengths": ["code", "reasoning", "general"],
        "daily_limit": 14400,
        "context_length": 128000,
        "speed": "fast",
    },
    "groq/llama-3.1-8b-instant": {
        "provider": "groq",
        "name": "Llama 3.1 8B Instant",
        "strengths": ["general", "quick"],
        "daily_limit": 14400,
        "context_length": 128000,
        "speed": "very_fast",
    },
    "groq/deepseek-r1-distill-llama-70b": {
        "provider": "groq",
        "name": "DeepSeek R1",
        "strengths": ["reasoning", "analysis", "math"],
        "daily_limit": 1000,
        "context_length": 128000,
        "speed": "medium",
    },
    "groq/gemma2-9b-it": {
        "provider": "groq",
        "name": "Gemma 2 9B",
        "strengths": ["multilingual", "general"],
        "daily_limit": 14400,
        "context_length": 8192,
        "speed": "fast",
    },
    # === OPENROUTER ===
    "openrouter/meta-llama/llama-4-maverick:free": {
        "provider": "openrouter",
        "name": "Llama 4 Maverick",
        "strengths": ["general", "creative", "complex"],
        "daily_limit": 50,
        "context_length": 128000,
        "speed": "medium",
    },
    "openrouter/mistralai/mistral-small-3.1-24b-instruct:free": {
        "provider": "openrouter",
        "name": "Mistral Small 3.1",
        "strengths": ["creative", "writing", "multilingual"],
        "daily_limit": 50,
        "context_length": 32000,
        "speed": "fast",
    },
    "openrouter/google/gemma-3-27b-it:free": {
        "provider": "openrouter",
        "name": "Gemma 3 27B",
        "strengths": ["general", "balanced"],
        "daily_limit": 50,
        "context_length": 8192,
        "speed": "medium",
    },
    # === GEMINI ===
    "gemini/gemini-2.0-flash": {
        "provider": "gemini",
        "name": "Gemini 2.0 Flash",
        "strengths": ["general", "long_context", "multimodal"],
        "daily_limit": 1000,
        "context_length": 1000000,
        "speed": "very_fast",
    },
    # === HUGGINGFACE ===
    "huggingface/microsoft/Phi-3-mini-4k-instruct": {
        "provider": "huggingface",
        "name": "Phi-3 Mini",
        "strengths": ["quick", "light", "code"],
        "hourly_limit": 300,
        "daily_limit": 7200,
        "context_length": 4096,
        "speed": "very_fast",
    },
    # === COHERE ===
    "cohere/command-r": {
        "provider": "cohere",
        "name": "Command R",
        "strengths": ["rag", "summarization", "analysis"],
        "monthly_limit": 1000,
        "daily_limit": 33,  # ~1000/30
        "context_length": 128000,
        "speed": "medium",
    },
    # === CLOUDFLARE ===
    "cloudflare/@cf/meta/llama-3.1-8b-instruct": {
        "provider": "cloudflare",
        "name": "Llama 3.1 8B (Edge)",
        "strengths": ["quick", "edge", "general"],
        "daily_limit": 10000,
        "context_length": 8192,
        "speed": "very_fast",
    },
}

# Mapping des types de tâches vers les forces des modèles
TASK_TYPE_MAPPING = {
    "code": ["code", "reasoning"],
    "creative": ["creative", "writing"],
    "analysis": ["reasoning", "analysis", "rag"],
    "math": ["math", "reasoning"],
    "summarization": ["summarization", "rag"],
    "translation": ["multilingual"],
    "general": ["general", "balanced"],
    "quick": ["quick", "light"],
}


settings = Settings()


def get_configured_providers() -> List[str]:
    """Retourne la liste des providers avec une clé API configurée"""
    configured = []
    if settings.groq_api_key:
        configured.append("groq")
    if settings.openrouter_api_key:
        configured.append("openrouter")
    if settings.gemini_api_key:
        configured.append("gemini")
    if settings.huggingface_api_key:
        configured.append("huggingface")
    if settings.cohere_api_key:
        configured.append("cohere")
    if settings.cloudflare_api_key and settings.cloudflare_account_id:
        configured.append("cloudflare")
    return configured


def get_available_model_ids() -> List[str]:
    """Retourne les IDs des modèles dont le provider a une clé configurée"""
    configured_providers = get_configured_providers()
    return [
        model_id
        for model_id, config in MODELS_CATALOG.items()
        if config["provider"] in configured_providers
    ]
