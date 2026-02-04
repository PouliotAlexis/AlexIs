"""
Agent Routeur - Sélectionne le meilleur modèle AI selon la tâche et les quotas disponibles
"""

import re
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from backend.config import MODELS_CATALOG, TASK_TYPE_MAPPING, get_available_model_ids
from backend.services.quota_manager import quota_manager


@dataclass
class ModelScore:
    """Score d'un modèle pour une tâche donnée"""

    model_id: str
    score: float
    reasons: List[str]
    remaining_quota: int


@dataclass
class RoutingResult:
    """Résultat de la sélection de modèle"""

    selected_model: str
    model_name: str
    provider: str
    task_type: str
    score: float
    reasons: List[str]
    alternatives: List[str]


class RouterAgent:
    """
    Agent intelligent pour sélectionner le meilleur modèle AI.
    Prend en compte le type de tâche, les quotas disponibles et les forces des modèles.
    """

    # Keywords pour détecter le type de tâche
    TASK_KEYWORDS = {
        "code": [
            r"\bcode\b",
            r"\bprogramm",
            r"\bpython\b",
            r"\bjavascript\b",
            r"\bjava\b",
            r"\bfonction\b",
            r"\bfunction\b",
            r"\bclass\b",
            r"\bclasse\b",
            r"\bbug\b",
            r"\bdebug\b",
            r"\bapi\b",
            r"\bsql\b",
            r"\bhtml\b",
            r"\bcss\b",
            r"\breact\b",
            r"\bvue\b",
            r"\bangular\b",
            r"\balgorithm",
            r"\bscript\b",
        ],
        "creative": [
            r"\b(écri|writ)",
            r"\bhistoire\b",
            r"\bstory\b",
            r"\bpoè[mt]",
            r"\bpoem\b",
            r"\bcréati",
            r"\bcreativ",
            r"\bimagin",
            r"\binvent",
            r"\bscénario\b",
            r"\bscript\b",
            r"\broman\b",
            r"\bnovel\b",
            r"\blyric",
            r"\bchanson\b",
        ],
        "analysis": [
            r"\banalys",
            r"\bexpliq",
            r"\bexplain",
            r"\bcompare",
            r"\bcompar",
            r"\bévaluer\b",
            r"\bevaluat",
            r"\brecherch",
            r"\bresearch",
            r"\bétudi",
            r"\bstud(y|ie)",
            r"\bcritiqu",
            r"\bexamin",
        ],
        "math": [
            r"\bmath",
            r"\bcalcul",
            r"\béquation\b",
            r"\bequation\b",
            r"\bformul",
            r"\bstatistiq",
            r"\bstatistic",
            r"\bprobabilit",
            r"\bgéométr",
            r"\bgeometr",
            r"\balgebr",
            r"\bintégral",
            r"\bintegral",
            r"\bdériv",
            r"\bderivat",
        ],
        "summarization": [
            r"\brésume",
            r"\bsummar",
            r"\bsynthès",
            r"\bsynthes",
            r"\bcondense",
            r"\bshorten",
            r"\brédu(is|ire)",
            r"\babridg",
        ],
        "translation": [
            r"\btradui",
            r"\btranslat",
            r"\bconvert",
            r"\bfrench\b",
            r"\benglish\b",
            r"\bespagnol\b",
            r"\bspanish\b",
            r"\ballemand\b",
            r"\bgerman\b",
        ],
        "quick": [
            r"\brapide",
            r"\bquick",
            r"\bfast\b",
            r"\bsimple\b",
            r"\bcourt\b",
            r"\bshort\b",
            r"\bbref\b",
            r"\bbrief\b",
        ],
    }

    def __init__(self):
        self._compiled_patterns = {
            task_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for task_type, patterns in self.TASK_KEYWORDS.items()
        }

    def detect_task_type(self, prompt: str) -> str:
        """
        Détecte le type de tâche à partir du prompt.

        Args:
            prompt: Le texte du prompt

        Returns:
            Le type de tâche détecté (code, creative, analysis, etc.)
        """
        scores = {}

        for task_type, patterns in self._compiled_patterns.items():
            score = sum(1 for p in patterns if p.search(prompt))
            if score > 0:
                scores[task_type] = score

        if not scores:
            return "general"

        # Retourner le type avec le plus haut score
        return max(scores, key=scores.get)

    def _calculate_model_score(
        self, model_id: str, task_type: str, available_only: bool = True
    ) -> Optional[ModelScore]:
        """
        Calcule le score d'un modèle pour une tâche donnée.

        Args:
            model_id: Identifiant du modèle
            task_type: Type de tâche détecté
            available_only: Si True, exclut les modèles sans quota

        Returns:
            ModelScore ou None si le modèle n'est pas disponible
        """
        if model_id not in MODELS_CATALOG:
            return None

        model_info = MODELS_CATALOG[model_id]
        remaining = quota_manager.get_remaining(model_id)

        # Vérifier la disponibilité
        if available_only and remaining <= 0:
            return None

        score = 0.0
        reasons = []

        # Score basé sur les forces du modèle vs le type de tâche
        required_strengths = TASK_TYPE_MAPPING.get(task_type, ["general"])
        model_strengths = model_info.get("strengths", [])

        matching_strengths = set(required_strengths) & set(model_strengths)
        strength_score = len(matching_strengths) * 20
        score += strength_score

        if matching_strengths:
            reasons.append(f"Forces correspondantes: {', '.join(matching_strengths)}")

        # Bonus pour le quota restant (favorise les modèles avec plus de quota)
        quota_ratio = min(remaining / 1000, 1.0)  # Normaliser
        quota_score = quota_ratio * 10
        score += quota_score
        reasons.append(f"Quota restant: {remaining}")

        # Bonus pour la vitesse si tâche "quick"
        if task_type == "quick":
            speed = model_info.get("speed", "medium")
            speed_bonus = {"very_fast": 15, "fast": 10, "medium": 5}.get(speed, 0)
            score += speed_bonus
            if speed_bonus > 5:
                reasons.append(f"Vitesse: {speed}")

        # Bonus pour le contexte long si prompt long
        context_length = model_info.get("context_length", 8000)
        if context_length >= 100000:
            score += 5
            reasons.append("Contexte long disponible")

        return ModelScore(
            model_id=model_id, score=score, reasons=reasons, remaining_quota=remaining
        )

    def select_model(self, prompt: str) -> RoutingResult:
        """
        Sélectionne le meilleur modèle pour un prompt donné.

        Args:
            prompt: Le texte du prompt

        Returns:
            RoutingResult avec le modèle sélectionné et les alternatives
        """
        # Détecter le type de tâche
        task_type = self.detect_task_type(prompt)

        # Scorer uniquement les modèles dont le provider a une clé configurée
        scores: List[ModelScore] = []
        configured_models = get_available_model_ids()
        for model_id in configured_models:
            model_score = self._calculate_model_score(model_id, task_type)
            if model_score:
                scores.append(model_score)

        if not scores:
            # Aucun modèle disponible - fallback
            return RoutingResult(
                selected_model="groq/llama-3.1-8b-instant",
                model_name="Llama 3.1 8B (Fallback)",
                provider="groq",
                task_type=task_type,
                score=0,
                reasons=[
                    "Aucun modèle avec quota disponible - utilisation du fallback"
                ],
                alternatives=[],
            )

        # Trier par score décroissant
        scores.sort(key=lambda s: s.score, reverse=True)

        best = scores[0]
        model_info = MODELS_CATALOG[best.model_id]

        # Alternatives (2ème et 3ème meilleurs)
        alternatives = [s.model_id for s in scores[1:4]]

        return RoutingResult(
            selected_model=best.model_id,
            model_name=model_info["name"],
            provider=model_info["provider"],
            task_type=task_type,
            score=best.score,
            reasons=best.reasons,
            alternatives=alternatives,
        )

    def get_model_for_optimization(self) -> str:
        """
        Retourne le modèle à utiliser pour l'optimisation de prompt.
        Choisit un modèle rapide avec quota disponible et clé API configurée.

        Returns:
            ID du modèle pour l'optimisation
        """
        # Modèles configurés (avec clé API)
        configured_models = set(get_available_model_ids())

        # Préférence pour les modèles rapides
        preferred = [
            "groq/llama-3.1-8b-instant",
            "groq/gemma2-9b-it",
            "cloudflare/@cf/meta/llama-3.1-8b-instruct",
            "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
            "gemini/gemini-2.0-flash",
        ]

        for model_id in preferred:
            if model_id in configured_models and quota_manager.is_available(model_id):
                return model_id

        # Fallback: premier modèle configuré et disponible
        for model_id in configured_models:
            if quota_manager.is_available(model_id):
                return model_id

        # Dernier recours
        return (
            list(configured_models)[0]
            if configured_models
            else "openrouter/mistralai/mistral-small-3.1-24b-instruct:free"
        )


# Instance globale
router_agent = RouterAgent()
