"""
Data Guard - Protection contre les fuites de données sensibles
Détecte et masque les informations sensibles avant envoi aux modèles AI
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class SensitiveDataType(Enum):
    """Types de données sensibles détectables"""

    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"  # Numéro d'assurance sociale
    IP_ADDRESS = "ip_address"
    DATE = "date"
    ADDRESS = "address"
    API_KEY = "api_key"
    PASSWORD = "password"
    JWT_TOKEN = "jwt_token"
    AWS_KEY = "aws_key"


@dataclass
class Detection:
    """Représente une détection de données sensibles"""

    data_type: SensitiveDataType
    original: str
    replacement: str
    start: int
    end: int


@dataclass
class ScanResult:
    """Résultat du scan de protection"""

    safe_prompt: str
    detections: List[Detection] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_safe: bool = True

    @property
    def detected_types(self) -> List[str]:
        return list(set(d.data_type.value for d in self.detections))

    @property
    def detection_count(self) -> int:
        return len(self.detections)


class DataGuard:
    """
    Service de protection des données sensibles.
    Scanne les prompts et masque les informations sensibles avant envoi aux APIs AI.
    """

    # Patterns de détection (regex)
    PATTERNS: Dict[SensitiveDataType, Tuple[str, str]] = {
        SensitiveDataType.EMAIL: (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL_REDACTED]",
        ),
        SensitiveDataType.PHONE: (
            r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}(?:[-.\s]?\d{2,4})?\b",
            "[PHONE_REDACTED]",
        ),
        SensitiveDataType.CREDIT_CARD: (
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "[CARD_REDACTED]",
        ),
        SensitiveDataType.SSN: (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "[SSN_REDACTED]"),
        SensitiveDataType.IP_ADDRESS: (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_REDACTED]"),
        SensitiveDataType.DATE: (
            r"\b(?:0[1-9]|[12][0-9]|3[01])[-./](?:0[1-9]|1[012])[-./](?:19|20)\d\d\b",
            "[DATE_REDACTED]",
        ),
        SensitiveDataType.ADDRESS: (
            r"\b\d+\s+[A-Za-z\s]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Rue|Alles|Lane|Ln|Drive|Dr)\b",
            "[ADDRESS_REDACTED]",
        ),
        SensitiveDataType.API_KEY: (
            r'\b(?:sk-[a-zA-Z0-9_\-]{20,}|api[_-]?key\s*[=:]\s*["\']?[a-zA-Z0-9_\-]{20,}["\']?)\b',
            "[API_KEY_REDACTED]",
        ),
        SensitiveDataType.PASSWORD: (
            r'(?:password|passwd|pwd|mot\s*de\s*passe)\s*[=:]\s*["\']?[^\s"\']{4,}["\']?',
            "[PASSWORD_REDACTED]",
        ),
        SensitiveDataType.JWT_TOKEN: (
            r"\beyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b",
            "[JWT_REDACTED]",
        ),
        SensitiveDataType.AWS_KEY: (
            r"\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b",
            "[AWS_KEY_REDACTED]",
        ),
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialise le Data Guard.

        Args:
            strict_mode: Si True, bloque les requêtes avec données sensibles.
                        Si False, masque et avertit seulement.
        """
        self.strict_mode = strict_mode
        self._compiled_patterns = {
            data_type: (re.compile(pattern, re.IGNORECASE), replacement)
            for data_type, (pattern, replacement) in self.PATTERNS.items()
        }

    def scan_and_protect(self, prompt: str) -> ScanResult:
        """
        Scanne le prompt et masque les données sensibles.

        Args:
            prompt: Le texte à analyser

        Returns:
            ScanResult avec le prompt sécurisé et les détections
        """
        detections: List[Detection] = []
        safe_prompt = prompt
        offset = 0

        # Scanner chaque type de données sensibles
        for data_type, (pattern, replacement) in self._compiled_patterns.items():
            for match in pattern.finditer(prompt):
                original = match.group()
                start = match.start()
                end = match.end()

                detections.append(
                    Detection(
                        data_type=data_type,
                        original=original,
                        replacement=replacement,
                        start=start,
                        end=end,
                    )
                )

        # Trier par position (inverse) pour remplacer de la fin vers le début
        detections.sort(key=lambda d: d.start, reverse=True)

        # Appliquer les remplacements
        for detection in detections:
            safe_prompt = (
                safe_prompt[: detection.start]
                + detection.replacement
                + safe_prompt[detection.end :]
            )

        # Remettre dans l'ordre chronologique pour le rapport
        detections.reverse()

        # Générer les warnings
        warnings = []
        if detections:
            types_found = set(d.data_type.value for d in detections)
            warnings.append(
                f"⚠️ {len(detections)} donnée(s) sensible(s) détectée(s) et masquée(s): "
                f"{', '.join(types_found)}"
            )

        return ScanResult(
            safe_prompt=safe_prompt,
            detections=detections,
            warnings=warnings,
            is_safe=len(detections) == 0 or not self.strict_mode,
        )

    def quick_check(self, prompt: str) -> bool:
        """
        Vérification rapide: retourne True si le prompt semble sûr.

        Args:
            prompt: Le texte à vérifier

        Returns:
            True si aucune donnée sensible détectée
        """
        for _, (pattern, _) in self._compiled_patterns.items():
            if pattern.search(prompt):
                return False
        return True


# Instance globale
data_guard = DataGuard(strict_mode=False)
