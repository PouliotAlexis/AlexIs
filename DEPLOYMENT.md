# 🚀 Guide de Déploiement

Ce guide explique comment mettre en ligne votre "AI WebSite" gratuitement.

## Option Recommandée : Hugging Face Spaces

C'est la méthode la plus simple et gratuite pour héberger des démos d'IA.

### 1. Préparer le projet
Assurez-vous d'avoir le fichier `Dockerfile` à la racine du projet (il a déjà été créé pour vous).

### 2. Créer un Space sur Hugging Face
1. Créez un compte sur [huggingface.co](https://huggingface.co/join) si ce n'est pas fait.
2. Allez sur [Nouveau Space](https://huggingface.co/new-space).
3. Remplissez les infos :
   - **Name** : `mon-ai-website` (ou ce que vous voulez)
   - **License** : `MIT` (ou autre)
   - **SDK** : Choisissez **Docker** (c'est important !)
   - **Space Hardware** : `CPU Basic (Free)` suffit pour commencer.
4. Cliquez sur **Create Space**.

### 3. Uploader le code
Une fois le Space créé, vous verrez des commandes git. Vous pouvez soit utiliser git en ligne de commande, soit uploader les fichiers directement via le navigateur.

**Méthode via le navigateur (plus simple) :**
1. Dans votre Space, allez dans l'onglet **Files**.
2. Cliquez sur **Add file > Upload files**.
3. Glissez-déposez TOUS les fichiers de votre dossier `AI_WebSite` (y compris `Dockerfile`, `requirements.txt`, les dossiers `backend` et `frontend`).
   > *Note : Ne pas uploader le dossier `.env` ou `__pycache__`.*
4. Cliquez sur **Commit changes**.

Le Space va automatiquement "Build" (construire) votre conteneur. Cela peut prendre 2-3 minutes.

### 4. Configurer les Clés API (Secrets)
Pour que votre site fonctionne, il a besoin de vos clés API (Groq, OpenRouter, etc.). **NE JAMAIS mettre vos clés dans le code public !**

1. Dans votre Space, allez dans l'onglet **Settings**.
2. Cherchez la section **Variables and secrets**.
3. Cliquez sur **New secret**.
4. Ajoutez vos clés une par une, exactement comme dans votre fichier `.env` local :
   - `GROQ_API_KEY`
   - `OPENROUTER_API_KEY`
   - `GEMINI_API_KEY`
   - etc.

### 5. Profitez !
Retournez sur l'onglet **App**. Si le build est fini (statut "Running" en vert), votre site est en ligne et accessible via l'URL du Space ! 🌍

---

## Option 2 : Exécution Locale avec Docker

Si vous avez Docker installé sur votre ordinateur :

1. **Construire l'image :**
   ```bash
   docker build -t ai-website .
   ```

2. **Lancer le conteneur :**
   ```bash
   docker run -p 7860:7860 --env-file .env ai-website
   ```

3. Ouvrez `http://localhost:7860` dans votre navigateur.

## 🌍 Accéder à votre Site

Une fois le déploiement terminé sur Hugging Face Spaces (Statut **Running**), vous avez deux façons d'accéder à votre site :

1.  **Via l'interface Hugging Face** :
    Allez sur l'onglet **App** de votre Space : `https://huggingface.co/spaces/PouliotAlexis/AlexIs`

2.  **Via le Lien Direct (Plein Écran)** :
    Votre site est accessible directement à cette adresse :
    👉 **[https://pouliotalexis-alexis.hf.space](https://pouliotalexis-alexis.hf.space)**

