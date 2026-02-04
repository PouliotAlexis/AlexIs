"""
Serveur FastAPI - API principale pour le site AI Générative Multi-Agents
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os

from backend.config import settings, MODELS_CATALOG
from backend.services.data_guard import data_guard
from backend.services.quota_manager import quota_manager
from backend.services.ai_service import ai_service
from backend.services.ocr_service import ocr_service
from backend.agents.router_agent import router_agent
from backend.agents.prompt_optimizer import prompt_optimizer


# === Modèles Pydantic ===


class Message(BaseModel):
    """Message de chat"""

    role: str
    content: str


class ImageData(BaseModel):
    """Image en base64"""

    name: str
    type: str
    base64: str


class GenerateRequest(BaseModel):
    """Requête de génération"""

    prompt: str = Field(..., min_length=1, max_length=100000)
    optimize_prompt: bool = Field(
        default=True, description="Optimiser le prompt avant envoi"
    )
    enable_data_guard: bool = Field(
        default=True, description="Activer la protection des données sensibles"
    )
    model_id: Optional[str] = Field(
        default=None, description="Forcer un modèle spécifique"
    )
    images: List[ImageData] = Field(
        default=[], description="Images en base64 pour analyse multimodale"
    )
    history: List[Message] = Field(
        default=[], description="Historique de la conversation"
    )
    max_tokens: int = Field(
        default=4096, description="Nombre maximum de tokens à générer"
    )
    language: str = Field(default="fr", description="Langue de l'interface (fr/en)")


class DataGuardInfo(BaseModel):
    """Infos sur la protection des données"""

    was_cleaned: bool
    detected_types: List[str]
    warnings: List[str]


class PromptOptimizationInfo(BaseModel):
    """Infos sur l'optimisation du prompt"""

    was_optimized: bool
    original: str
    optimized: str
    optimization_model: Optional[str]
    improvements: List[str]


class ModelInfo(BaseModel):
    """Infos sur le modèle utilisé"""

    model_id: str
    model_name: str
    provider: str
    task_type: str
    selection_reasons: List[str]


class GenerateResponse(BaseModel):
    """Réponse de génération"""

    success: bool
    content: str
    model: ModelInfo
    data_guard: DataGuardInfo
    optimization: PromptOptimizationInfo
    error: Optional[str] = None


class QuotaStatus(BaseModel):
    """Statut des quotas d'un modèle"""

    model_id: str
    name: str
    provider: str
    usage: int
    daily_limit: int
    remaining: int
    available: bool
    percentage_used: float


class HealthResponse(BaseModel):
    """Réponse de santé du serveur"""

    status: str
    available_models: int
    total_models: int


# === Application FastAPI ===

app = FastAPI(
    title="AI Générative Multi-Agents",
    description="API pour générer du contenu via plusieurs modèles AI avec sélection intelligente",
    version="1.0.0",
)

# CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Endpoints ===


@app.on_event("startup")
async def startup_event():
    """Tâches au démarrage"""
    # Précharger les modèles OCR en arrière-plan
    import threading

    threading.Thread(target=ocr_service.preload_models, daemon=True).start()


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état du serveur"""
    available = quota_manager.get_available_models()
    return HealthResponse(
        status="healthy",
        available_models=len(available),
        total_models=len(MODELS_CATALOG),
    )


@app.get("/api/models", response_model=List[QuotaStatus])
async def get_models():
    """Liste tous les modèles avec leur statut de quota"""
    status = quota_manager.get_all_status()
    return [QuotaStatus(model_id=model_id, **info) for model_id, info in status.items()]


@app.get("/api/models/available", response_model=List[str])
async def get_available_models():
    """Liste les modèles encore disponibles (quota non épuisé)"""
    return quota_manager.get_available_models()


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Point d'entrée principal pour la génération.

    1. Protège les données sensibles (Data Guard)
    2. Sélectionne le meilleur modèle (Agent Routeur)
    3. Optimise le prompt si demandé (Agent Optimiseur)
    4. Génère la réponse (avec historique)
    """
    try:
        # === 1. Data Guard ===
        safe_prompt = request.prompt
        data_guard_info = DataGuardInfo(
            was_cleaned=False, detected_types=[], warnings=[]
        )

        if request.enable_data_guard:
            scan_result = data_guard.scan_and_protect(request.prompt)
            safe_prompt = scan_result.safe_prompt
            data_guard_info = DataGuardInfo(
                was_cleaned=scan_result.detection_count > 0,
                detected_types=scan_result.detected_types,
                warnings=scan_result.warnings,
            )
        else:
            # Si Data Guard désactivé, on ajoute juste un warning informatif
            data_guard_info.warnings = [
                "⚠️ Protection des données désactivée par l'utilisateur"
            ]

        # === 2. Sélection du modèle ===
        task_type = "general"
        selection_reasons = []
        routing_result = None

        if request.model_id and request.model_id in MODELS_CATALOG:
            # Modèle forcé par l'utilisateur
            if not quota_manager.is_available(request.model_id):
                raise HTTPException(
                    status_code=429,
                    detail=f"Quota épuisé pour le modèle {request.model_id}",
                )
            selected_model = request.model_id
            model_info = MODELS_CATALOG[selected_model]
            routing_result = None
            task_type = router_agent.detect_task_type(safe_prompt)
            selection_reasons = ["Modèle sélectionné manuellement par l'utilisateur"]
        else:
            # Sélection automatique par l'agent routeur
            routing_result = router_agent.select_model(safe_prompt)
            selected_model = routing_result.selected_model
            model_info = MODELS_CATALOG.get(selected_model, {})
            task_type = routing_result.task_type
            selection_reasons = routing_result.reasons

        model_response_info = ModelInfo(
            model_id=selected_model,
            model_name=model_info.get("name", "Unknown"),
            provider=model_info.get("provider", "unknown"),
            task_type=task_type,
            selection_reasons=selection_reasons,
        )

        # === 3. Optimisation du prompt ===
        if request.optimize_prompt:
            opt_result = await prompt_optimizer.optimize(
                safe_prompt, language=request.language
            )
            final_prompt = opt_result.optimized
            optimization_info = PromptOptimizationInfo(
                was_optimized=opt_result.was_optimized,
                original=safe_prompt,
                optimized=opt_result.optimized,
                optimization_model=opt_result.optimization_model,
                improvements=opt_result.improvements,
            )
        else:
            final_prompt = safe_prompt
            optimization_info = PromptOptimizationInfo(
                was_optimized=False,
                original=safe_prompt,
                optimized=safe_prompt,
                optimization_model=None,
                improvements=["Optimisation désactivée par l'utilisateur"],
            )

        # === 4. Génération avec fallback ===
        # Convert images to dict for the service
        images_data = [
            {"name": img.name, "type": img.type, "base64": img.base64}
            for img in request.images
        ]
        has_images = len(images_data) > 0

        # Si on a des images, forcer Gemini en premier
        if has_images:
            models_to_try = ["gemini/gemini-2.0-flash"]
            # Ajouter les alternatives textuelles
            if routing_result and routing_result.alternatives:
                models_to_try.extend(routing_result.alternatives)
            else:
                models_to_try.extend(
                    [selected_model]
                    if selected_model != "gemini/gemini-2.0-flash"
                    else []
                )
                models_to_try.extend(
                    [
                        "groq/llama-3.3-70b-versatile",
                        "openrouter/meta-llama/llama-4-maverick:free",
                    ]
                )
        else:
            models_to_try = [selected_model]
            if routing_result and routing_result.alternatives:
                models_to_try.extend(routing_result.alternatives)

        response = None
        used_model = selected_model
        fallback_used = False
        images_converted_to_text = False

        for i, try_model in enumerate(models_to_try):
            # Pour le premier modèle (Gemini) avec images, envoyer les images
            # Pour les autres modèles, convertir les images en texte
            if has_images and i == 0:
                # Essayer Gemini avec les images
                response = await ai_service.generate(
                    try_model,
                    final_prompt,
                    images=images_data,
                    history=request.history,
                    max_tokens=request.max_tokens,
                )
            elif has_images and i > 0 and not images_converted_to_text:
                # Fallback: extraire le texte des images via OCR
                extracted_text = await ocr_service.extract_text_from_images(images_data)
                final_prompt = f"{final_prompt}\n\n--- Texte extrait des images (OCR) ---\n{extracted_text}"
                images_converted_to_text = True
                response = await ai_service.generate(
                    try_model,
                    final_prompt,
                    history=request.history,
                    max_tokens=request.max_tokens,
                )
            else:
                response = await ai_service.generate(
                    try_model,
                    final_prompt,
                    history=request.history,
                    max_tokens=request.max_tokens,
                )

            if response.success:
                used_model = try_model
                break
            else:
                # Si erreur 429 ou autre erreur API, essayer le suivant
                if (
                    "429" in str(response.error)
                    or "quota" in str(response.error).lower()
                ):
                    fallback_used = True
                    continue
                else:
                    # Autre erreur, on continue quand même
                    fallback_used = True
                    continue

        # Mettre à jour les infos du modèle si fallback utilisé
        if fallback_used and used_model != selected_model:
            model_info = MODELS_CATALOG.get(used_model, {})
            fallback_notes = [f"Fallback depuis {selected_model}"]
            if images_converted_to_text:
                fallback_notes.append(
                    "Images converties en texte (Gemini non disponible)"
                )
            model_response_info = ModelInfo(
                model_id=used_model,
                model_name=model_info.get("name", "Unknown"),
                provider=model_info.get("provider", "unknown"),
                task_type=task_type,
                selection_reasons=selection_reasons + fallback_notes,
            )

        # Incrémenter le quota du modèle utilisé
        quota_manager.increment_usage(used_model)

        if not response or not response.success:
            return GenerateResponse(
                success=False,
                content="",
                model=model_response_info,
                data_guard=data_guard_info,
                optimization=optimization_info,
                error=response.error if response else "Tous les modèles ont échoué",
            )

        return GenerateResponse(
            success=True,
            content=response.content,
            model=model_response_info,
            data_guard=data_guard_info,
            optimization=optimization_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scan")
async def scan_prompt(prompt: str):
    """Scanne un prompt pour détecter les données sensibles (sans générer)"""
    result = data_guard.scan_and_protect(prompt)
    return {
        "is_safe": result.is_safe,
        "detected_types": result.detected_types,
        "warnings": result.warnings,
        "safe_prompt": result.safe_prompt,
        "detection_count": result.detection_count,
    }


# === Servir le frontend ===

frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_frontend():
        """Sert la page d'accueil"""
        return FileResponse(os.path.join(frontend_path, "index.html"))


# === Point d'entrée ===

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
