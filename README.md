---
title: AI Website
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
login_required: true
---

# 🧠 AI Générative Multi-Agents

Une plateforme web intelligente qui utilise une architecture multi-agents pour optimiser vos interactions avec les Grands Modèles de Langage (LLMs). Le système sélectionne automatiquement le meilleur modèle pour votre tâche, sécurise vos données et améliore vos prompts.

![AI WebSite Screenshot](https://via.placeholder.com/800x400?text=Interface+Utilisateur+AI+Multi-Agents)

## ✨ Fonctionnalités Clés

- **🔄 Routage Intelligent** : Analyse votre demande et sélectionne l'IA la plus compétente (Code, Créativité, Analyse...) et la plus économique.
- **🛡️ Data Guard (Protection des Données)** : Détecte et masque automatiquement les informations sensibles (emails, téléphones, clés API, cartes bancaires...) *avant* l'envoi aux serveurs tiers.
- **✨ Optimisation de Prompt** : Un agent spécialisé réécrit vos prompts pour maximiser la qualité des réponses.
- **⚡ Fallback Automatique** : Si une IA est indisponible ou surchargée, le système bascule instantanément sur un modèle de secours sans interruption.
- **👁️ Analyse Multimodale** : Supporte l'analyse d'images via Gemini Vision ou OCR automatique si nécessaire.
- **🎨 Interface Premium** : Une UI moderne, fluide et responsive avec mode sombre et effets glassmorphism.

## 🏗️ Architecture Multi-Agents

Le système repose sur un pipeline de 4 agents spécialisés :

1.  **Le Gardien (Data Guard)** : Filtre les données personnelles (PII) via Regex.
2.  **Le Stratège (Router Agent)** : Score les modèles disponibles selon la tâche détectée.
3.  **Le Rédacteur (Prompt Optimizer)** : Reformule la requête pour plus de clarté.
4.  **L'Orchestrateur** : Gère l'exécution, les quotas et les erreurs.

## 🚀 Installation

### Prérequis
- Python 3.8+
- Un navigateur web moderne

### Configuration

1.  **Cloner le dépôt**
    ```bash
    git clone https://github.com/votre-user/ai-website.git
    cd ai-website
    ```

2.  **Créer un environnement virtuel**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurer les variables d'environnement**
    Copiez le fichier d'exemple et ajoutez vos clés API :
    ```bash
    cp .env.example .env
    ```
    Éditez `.env` avec vos clés (Groq, Gemini, OpenRouter, HuggingFace, etc.).

### Lancement

1.  **Démarrer le serveur Backend**
    ```bash
    python -m uvicorn backend.server:app --reload
    ```

2.  **Accéder à l'application**
    Ouvrez votre navigateur sur : `http://localhost:8000`

## 🛠️ Stack Technique

- **Backend** : FastAPI (Python), Pydantic, httpx
- **Frontend** : HTML5, CSS3 (Variables, Flexbox/Grid), JavaScript (Vanilla)
- **IA** : Intégration de multiples providers (Groq, Google Gemini, OpenRouter, HuggingFace, Cloudflare)
- **Outils** : Uvicorn, Python-Dotenv

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
