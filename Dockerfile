# Utiliser une image Python officielle légère
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 user

# Donner les permissions à l'utilisateur sur le dossier de travail
RUN chown user:user /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copier les fichiers de dépendances
COPY --chown=user requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY --chown=user . .

# Exposer le port 7860 (standard pour Hugging Face Spaces)
EXPOSE 7860

# Commande de démarrage
# On écoute sur 0.0.0.0 pour être accessible depuis l'extérieur du conteneur
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860"]
