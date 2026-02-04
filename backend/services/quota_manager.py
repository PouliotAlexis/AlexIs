"""
Gestionnaire de Quotas - Track les requêtes par modèle et gère les limites
"""

import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from backend.config import MODELS_CATALOG


@dataclass
class ModelUsage:
    """Usage d'un modèle pour une journée"""

    model_id: str
    count: int
    date: str  # Format: YYYY-MM-DD

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelUsage":
        return cls(**data)


class QuotaManager:
    """
    Gestionnaire des quotas d'utilisation des modèles AI.
    Persiste les données dans un fichier JSON.
    Reset automatique quotidien.
    """

    def __init__(self, data_path: str = "data/quotas.json"):
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self._usage: Dict[str, ModelUsage] = {}
        self._load()

    def _today(self) -> str:
        """Retourne la date d'aujourd'hui au format YYYY-MM-DD"""
        return date.today().isoformat()

    def _load(self) -> None:
        """Charge les données depuis le fichier JSON"""
        if self.data_path.exists():
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for model_id, usage_data in data.items():
                        self._usage[model_id] = ModelUsage.from_dict(usage_data)
            except (json.JSONDecodeError, KeyError):
                self._usage = {}

        # Reset si la date a changé
        self._check_daily_reset()

    def _save(self) -> None:
        """Sauvegarde les données dans le fichier JSON"""
        data = {model_id: usage.to_dict() for model_id, usage in self._usage.items()}
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _check_daily_reset(self) -> None:
        """Vérifie et effectue le reset quotidien si nécessaire"""
        today = self._today()
        reset_needed = False

        for model_id, usage in self._usage.items():
            if usage.date != today:
                reset_needed = True
                break

        if reset_needed:
            self._usage = {}
            self._save()

    def get_usage(self, model_id: str) -> int:
        """
        Retourne le nombre de requêtes effectuées aujourd'hui pour un modèle.

        Args:
            model_id: Identifiant du modèle

        Returns:
            Nombre de requêtes effectuées
        """
        self._check_daily_reset()

        if model_id in self._usage:
            return self._usage[model_id].count
        return 0

    def get_remaining(self, model_id: str) -> int:
        """
        Retourne le nombre de requêtes restantes pour un modèle.

        Args:
            model_id: Identifiant du modèle

        Returns:
            Nombre de requêtes restantes
        """
        if model_id not in MODELS_CATALOG:
            return 0

        daily_limit = MODELS_CATALOG[model_id].get("daily_limit", 0)
        usage = self.get_usage(model_id)
        return max(0, daily_limit - usage)

    def increment_usage(self, model_id: str, count: int = 1) -> None:
        """
        Incrémente le compteur d'utilisation d'un modèle.

        Args:
            model_id: Identifiant du modèle
            count: Nombre à ajouter (défaut: 1)
        """
        self._check_daily_reset()
        today = self._today()

        if model_id in self._usage:
            self._usage[model_id].count += count
        else:
            self._usage[model_id] = ModelUsage(
                model_id=model_id, count=count, date=today
            )

        self._save()

    def is_available(self, model_id: str) -> bool:
        """
        Vérifie si un modèle a encore du quota disponible.

        Args:
            model_id: Identifiant du modèle

        Returns:
            True si le modèle peut encore être utilisé
        """
        return self.get_remaining(model_id) > 0

    def get_available_models(self) -> List[str]:
        """
        Retourne la liste des modèles encore disponibles.

        Returns:
            Liste des IDs de modèles avec quota restant
        """
        return [
            model_id
            for model_id in MODELS_CATALOG.keys()
            if self.is_available(model_id)
        ]

    def get_all_status(self) -> Dict[str, dict]:
        """
        Retourne le statut de tous les modèles.

        Returns:
            Dict avec usage, limite et restant pour chaque modèle
        """
        self._check_daily_reset()
        status = {}

        for model_id, model_info in MODELS_CATALOG.items():
            daily_limit = model_info.get("daily_limit", 0)
            usage = self.get_usage(model_id)
            remaining = max(0, daily_limit - usage)

            status[model_id] = {
                "name": model_info["name"],
                "provider": model_info["provider"],
                "usage": usage,
                "daily_limit": daily_limit,
                "remaining": remaining,
                "available": remaining > 0,
                "percentage_used": round(
                    (usage / daily_limit * 100) if daily_limit > 0 else 0, 1
                ),
            }

        return status


# Instance globale
quota_manager = QuotaManager()
