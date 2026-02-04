"""
Service OCR (Optical Character Recognition)
Extrait le texte des images quand Gemini n'est pas disponible
"""

import base64
import io
import asyncio
from typing import List, Dict
from PIL import Image
from functools import partial


class OCRService:
    """Service d'extraction de texte depuis les images"""

    def __init__(self):
        self._reader = None
        self._is_loading = False

    def _get_reader(self):
        """Lazy loading du reader OCR (car il est lourd)"""
        if self._reader is None:
            import easyocr

            # Supporter français et anglais
            self._reader = easyocr.Reader(["fr", "en"], gpu=False)
        return self._reader

    def preload_models(self):
        """Précharge les modèles en mémoire (à appeler au démarrage)"""
        if self._reader is None and not self._is_loading:
            self._is_loading = True
            try:
                print("⏳ Chargement des modèles OCR en arrière-plan...")
                self._get_reader()
                print("✅ Modèles OCR chargés !")
            except Exception as e:
                print(f"❌ Erreur chargement OCR: {e}")
            finally:
                self._is_loading = False

    def extract_text_from_base64_sync(self, base64_data: str) -> str:
        """Version synchrone de l'extraction"""
        try:
            # Décoder le base64 en bytes
            image_bytes = base64.b64decode(base64_data)

            # Convertir en image PIL
            image = Image.open(io.BytesIO(image_bytes))

            # Convertir en RGB si nécessaire
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Sauvegarder temporairement
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name, "JPEG")
                tmp_path = tmp.name

            try:
                reader = self._get_reader()
                results = reader.readtext(tmp_path)
                extracted_texts = [
                    text for (_, text, confidence) in results if confidence > 0.3
                ]
                text = "\n".join(extracted_texts)
                return text if text.strip() else "[Aucun texte détecté dans l'image]"
            finally:
                # Nettoyer
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
        except Exception as e:
            return f"[Erreur OCR: {str(e)}]"

    async def extract_text_from_images(self, images: List[Dict]) -> str:
        """
        Extrait le texte de plusieurs images (Asynchrone)
        """
        if not images:
            return ""

        results = []
        for img in images:
            # Exécuter l'OCR (CPU-bound) dans un thread séparé
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None, partial(self.extract_text_from_base64_sync, img.get("base64", ""))
            )
            results.append(
                f"--- Texte extrait de {img.get('name', 'image')} ---\n{text}"
            )

        return "\n\n".join(results)


# Singleton
ocr_service = OCRService()
