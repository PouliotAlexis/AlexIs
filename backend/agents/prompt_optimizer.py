"""
Agent Optimiseur de Prompt - Améliore les prompts avant envoi aux modèles AI
"""

from typing import Optional
from dataclasses import dataclass

from backend.services.ai_service import ai_service
from backend.services.quota_manager import quota_manager
from backend.agents.router_agent import router_agent


@dataclass
class OptimizedPrompt:
    """Résultat de l'optimisation du prompt"""

    original: str
    optimized: str
    was_optimized: bool
    optimization_model: Optional[str] = None
    improvements: list = None

    def __post_init__(self):
        if self.improvements is None:
            self.improvements = []


class PromptOptimizer:
    """
    Agent pour améliorer les prompts avant envoi aux modèles AI.
    Utilise un modèle léger pour restructurer et clarifier les prompts.
    """

    # Prompt système pour l'optimisation
    PROMPTS = {
        "fr": """Tu es un expert en prompt engineering. Ta tâche est d'améliorer le prompt utilisateur pour obtenir de meilleures réponses d'un modèle AI.

Règles d'optimisation:
1. Garde l'intention originale intacte
2. Ajoute de la clarté et de la structure si nécessaire
3. Précise le format de réponse attendu si pas spécifié
4. Reste concis - n'ajoute pas de contenu inutile
5. Si le prompt est déjà clair et bien structuré, retourne-le tel quel

IMPORTANT: Retourne UNIQUEMENT le prompt amélioré, sans explication ni commentaire.""",
        "en": """You are an expert in prompt engineering. Your task is to improve the user prompt to get better responses from an AI model.

Optimization rules:
1. Keep the original intent intact
2. Add clarity and structure if necessary
3. Specify the expected response format if not specified
4. Keep it concise - do not add unnecessary content
5. If the prompt is already clear and well-structured, return it as is

IMPORTANT: Return ONLY the improved prompt, without explanation or commentary.""",
    }

    def __init__(self, min_length_to_optimize: int = 20, max_length_to_skip: int = 500):
        """
        Initialise l'optimiseur.

        Args:
            min_length_to_optimize: Longueur minimale du prompt pour optimiser
            max_length_to_skip: Longueur maximale - au-delà, on n'optimise pas (coûteux)
        """
        self.min_length = min_length_to_optimize
        self.max_length = max_length_to_skip

    def _should_optimize(self, prompt: str) -> bool:
        """
        Détermine si un prompt devrait être optimisé.

        Args:
            prompt: Le prompt à évaluer

        Returns:
            True si l'optimisation est recommandée
        """
        # Trop court = pas besoin
        if len(prompt) < self.min_length:
            return False

        # Trop long = trop coûteux
        if len(prompt) > self.max_length:
            return False

        # Vérifier qu'on a un modèle disponible pour optimiser
        opt_model = router_agent.get_model_for_optimization()
        if not quota_manager.is_available(opt_model):
            return False

        return True

    async def optimize(
        self, prompt: str, force: bool = False, language: str = "fr"
    ) -> OptimizedPrompt:
        """
        Optimise un prompt pour de meilleures réponses.

        Args:
            prompt: Le prompt original
            force: Si True, optimise même si les conditions ne sont pas remplies

        Returns:
            OptimizedPrompt avec le résultat
        """
        # Vérifier si on doit optimiser
        if not force and not self._should_optimize(prompt):
            return OptimizedPrompt(
                original=prompt,
                optimized=prompt,
                was_optimized=False,
                improvements=[
                    "Prompt utilisé tel quel (conditions d'optimisation non remplies)"
                ],
            )

        # Sélectionner le modèle d'optimisation
        opt_model = router_agent.get_model_for_optimization()

        # Construire le prompt d'optimisation
        system_prompt = self.PROMPTS.get(language, self.PROMPTS["fr"])
        optimization_request = f"""{system_prompt}

Prompt à améliorer:
\"\"\"
{prompt}
\"\"\"

Prompt amélioré:"""

        try:
            # Appeler le modèle
            response = await ai_service.generate(
                model_id=opt_model,
                prompt=optimization_request,
                temperature=0.3,  # Basse température pour plus de cohérence
                max_tokens=len(prompt) * 2,  # Limiter la réponse
            )

            # Incrémenter le quota
            quota_manager.increment_usage(opt_model)

            if response.success and response.content.strip():
                optimized = response.content.strip()

                # Nettoyer les guillemets si présents
                if optimized.startswith('"') and optimized.endswith('"'):
                    optimized = optimized[1:-1]
                if optimized.startswith("'") and optimized.endswith("'"):
                    optimized = optimized[1:-1]

                # Identifier les améliorations
                improvements = self._identify_improvements(prompt, optimized)

                return OptimizedPrompt(
                    original=prompt,
                    optimized=optimized,
                    was_optimized=True,
                    optimization_model=opt_model,
                    improvements=improvements,
                )
            else:
                # Échec de l'optimisation - utiliser l'original
                return OptimizedPrompt(
                    original=prompt,
                    optimized=prompt,
                    was_optimized=False,
                    optimization_model=opt_model,
                    improvements=[
                        f"Optimisation échouée: {response.error or 'réponse vide'}"
                    ],
                )

        except Exception as e:
            return OptimizedPrompt(
                original=prompt,
                optimized=prompt,
                was_optimized=False,
                improvements=[f"Erreur lors de l'optimisation: {str(e)}"],
            )

    def _identify_improvements(self, original: str, optimized: str) -> list:
        """
        Identifie les types d'améliorations apportées.

        Args:
            original: Prompt original
            optimized: Prompt optimisé

        Returns:
            Liste des améliorations détectées
        """
        improvements = []

        # Vérifier si c'est identique
        if original.strip() == optimized.strip():
            improvements.append("Prompt déjà optimal - aucun changement")
            return improvements

        # Longueur
        len_diff = len(optimized) - len(original)
        if len_diff > 20:
            improvements.append("Ajout de contexte/clarifications")
        elif len_diff < -20:
            improvements.append("Simplification et concision")

        # Structure
        if optimized.count("\n") > original.count("\n"):
            improvements.append("Meilleure structure (paragraphes)")

        # Ponctuation
        if optimized.count("?") > original.count("?"):
            improvements.append("Questions clarifiées")

        # Si aucune amélioration spécifique détectée
        if not improvements:
            improvements.append("Reformulation pour plus de clarté")

        return improvements


# Instance globale
prompt_optimizer = PromptOptimizer()
