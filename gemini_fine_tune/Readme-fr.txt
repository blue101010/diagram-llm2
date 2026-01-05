Problème : Si le fine-tuning n'est plus disponible via l'API Gemini standard (hors Vertex AI), 
le code d'exemple actuel (perform_inference.py) peut être trompeur car il utilise `genai.Client`.
Pour lancer un fine‑tuning Gemini en 2026, il faut utiliser le client Vertex AI.


Nouveau script : `perform_inference_vertex.py`
Ce script a été ajouté pour montrer comment interagir avec les modèles via Google Cloud Vertex AI en utilisant le nouveau SDK unifié `google-genai`.
Il nécessite une configuration Google Cloud (Projet ID, Location).

Différences clés :
- `perform_inference.py` : Utilise l'ancienne méthode ou l'API Key simple.
- `perform_inference_vertex.py` : Utilise le SDK `google-genai` avec le paramètre `vertexai=True`. C'est la méthode recommandée par Google pour remplacer l'ancien SDK `vertexai` déprécié. Elle permet d'accéder aux modèles Vertex AI (y compris les modèles fine-tunés) avec une authentification d'entreprise.

Comment exécuter le script :
Assurez-vous d'être à la racine du projet et que votre environnement virtuel est configuré.

Pour PowerShell (Windows) :
```powershell
.\venv\Scripts\python.exe gemini_fine_tune\perform_inference_vertex.py
```

Pour Bash (Git Bash) :
```bash
./venv/Scripts/python gemini_fine_tune/perform_inference_vertex.py
```

## Configuration de l'authentification (Obligatoire)

Pour que le script fonctionne, vous devez authentifier votre environnement local avec Google Cloud.

1.  **Installer Google Cloud CLI** :
    *   **Windows (PowerShell)** : Exécutez `.\install_gcloud.ps1`.
    *   **Windows (Bash)** : Si `gcloud` est introuvable, exécutez `source ./fix_gcloud_bash.sh`.
    *   **Autre** : https://cloud.google.com/sdk/docs/install
2.  **S'authentifier** : Ouvrez un terminal et lancez la commande suivante :
    ```bash
    gcloud auth application-default login
    ```
    Cela ouvrira une fenêtre de navigateur pour vous connecter à votre compte Google.
3.  **Définir le projet par défaut** (Optionnel mais recommandé) :
    ```bash
    gcloud config set project VOTRE_PROJECT_ID
    ```
    (Remplacez `VOTRE_PROJECT_ID` par l'ID de votre projet, ex: `1091107469352`)
