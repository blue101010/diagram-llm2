# Tutoriel et Plan de Test : Projet Diagram-LLM2

Ce document sert de guide complet pour comprendre, installer et tester le projet **Diagram-LLM2**. Ce projet vise à créer des jeux de données synthétiques de haute qualité pour entraîner des modèles d'IA (comme Gemini,gemma-3-27b ou Phi-4) à générer des diagrammes **Mermaid.js**.

---

## 1. Introduction

Le but de ce projet est d'automatiser la création de données d'entraînement pour les diagrammes. Au lieu d'écrire manuellement des milliers d'exemples, nous utilisons un LLM puissant (Gemini, Gemma, Phi) pour générer des exemples variés basés sur la documentation officielle de Mermaid.

Le flux de travail complet comprend :
1.  La génération de questions/réponses synthétiques.
2.  La validation technique du code Mermaid généré.
3.  La préparation des données pour le fine-tuning.
4.  La visualisation des résultats.

---

## 2. Comment ça fonctionne

Le système est modulaire :

*   **`synthetic_dataset_generator/`** : C'est le cœur du système. Il lit des fichiers Markdown contenant la documentation de Mermaid (dans `md/`) et utilise Gemini pour inventer des scénarios réalistes et le code Mermaid correspondant.
*   **`validator/`** : Une application Web locale (HTML/JS) qui utilise la librairie officielle `mermaid.js` pour tester si le code généré produit réellement un diagramme valide ou s'il contient des erreurs de syntaxe.
*   **`gemini_fine_tune/`** : Contient les scripts pour convertir les données brutes (JSON) en format JSONL optimisé pour le fine-tuning sur Google AI Studio, et pour lancer l'inférence.
*   **`previewer/`** : Une interface Web simple pour naviguer humainement dans le jeu de données et vérifier la qualité des "prompts".

---

## 3. Prérequis

Avant de commencer, assurez-vous d'avoir :

1.  **Python 3.10 ** ou supérieur installé.
2.  Une **Clé API Google Gemini**.
    *   Créez un fichier `.env` à la racine du projet (copiez `.env.example` s'il existe).
    *   Ajoutez votre clé : `GEMINI_API_KEY=votre_clé_ici`.
3.  Un navigateur web moderne (Chrome, Edge, Firefox) pour les outils de validation.

---

## 4. Tests et Usages (Plan de Test)

Suivez ces étapes pour tester l'ensemble du pipeline.

### Étape 1 : Installation de l'environnement

Ouvrez votre terminal à la racine du projet :

```bash
# 1. Créer un environnement virtuel
python -m venv venv

# 2. Activer l'environnement
# Sur Windows :
venv\Scripts\activate
# Sur Mac/Linux :
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Étape 2 : Génération de données synthétiques

Nous allons générer un petit lot de données pour tester.

1.  Allez dans le dossier du générateur :
    ```bash
    cd synthetic_dataset_generator
    ```
2.  Lancez le script principal :
    ```bash
    python main.py
    ```
3.  **Vérification** :
    *   Le script doit afficher des logs de progression.
    *   À la fin, vérifiez le dossier `../generated_synthetic_datasets/`. Vous devriez y trouver un fichier (ex: `raw_main.json` ou similaire selon la configuration).

### Étape 3 : Validation des données (Critique)

Les LLM font des erreurs. Nous devons filtrer le code Mermaid invalide.

> **Exemple d'erreur courante (Architecture Beta) :**
> Il est normal de rencontrer des erreurs de syntaxe, notamment sur les fonctionnalités expérimentales.
>
> *   **Code généré** :
>     ```mermaid
>     architecture-beta
>         service api(server)[API Gateway]
>         service auth(server)[Auth Service]
>         service users(database)[User DB]
>
>         api:B -- T:auth
>         api:R -- L:users
>     ```
> *   **Erreur** : `Syntax error in text`
>
> Le validateur détectera et exclura automatiquement ces cas du jeu de données final.

1.  Ouvrez le fichier `validator/validator_app.html` dans votre navigateur (double-cliquez dessus ou faites un glisser-déposer).
2.  **Chargement** : Cliquez sur le bouton pour charger un fichier ou glissez le fichier JSON généré à l'étape 2 dans la zone dédiée.
3.  **Traitement** : Cliquez sur **"Process Diagrams"**.
    *   L'outil va rendre chaque diagramme en arrière-plan.
    *   Il séparera les diagrammes valides des invalides.
4.  **Export** :
    *   Examinez l'onglet "Invalid Diagrams" pour voir les erreurs courantes.
    *   Cliquez sur **"Download Valid Diagrams"** pour récupérer le fichier propre (ex: `mermaid_valid_diagrams_....json`).

### Étape 4 : Préparation pour le Fine-Tuning

Transformons les données validées en format d'entraînement.

1.  Placez votre fichier validé (téléchargé à l'étape 3) dans un endroit accessible, ou utilisez le fichier brut pour ce test.
2.  Allez dans le dossier de fine-tuning :
    ```bash
    cd ../gemini_fine_tune
    ```
3.  Lancez la conversion :
    ```bash
    python convert_to_jsonl.py
    ```
    *(Note : Assurez-vous que le script pointe vers le bon fichier d'entrée dans son code ou via arguments, par défaut il cherche souvent dans `generated_synthetic_datasets`)*.
4.  **Vérification** : Regardez dans le dossier `dataset/`. Vous devriez voir `training_data.jsonl` et `validation_data.jsonl`.

### Étape 5 : Visualisation (Optionnel)

Pour inspecter visuellement ce que vous allez donner à manger à l'IA.

1.  Copiez un fichier JSON de données (ex: `raw_main.json`) dans le dossier `previewer/` et renommez-le `dataset.json`.
2.  Lancez un serveur local pour éviter les problèmes de sécurité (CORS) :
    ```bash
    python -m http.server 5000 --directory previewer
    ```
3.  Ouvrez votre navigateur à l'adresse : [http://localhost:5000](http://localhost:5000).
4.  Vous pouvez maintenant faire défiler les cartes montrant le "Prompt" (la demande utilisateur) et le "Diagramme" (le résultat visuel).

---

## Résolution de problèmes courants

*   **Erreur API Key** : Vérifiez que la variable d'environnement `GEMINI_API_KEY` est bien définie dans votre terminal ou votre fichier `.env`.
*   **Erreur CSP dans le navigateur** : Si le validateur ne charge pas Mermaid, vérifiez la console du navigateur (F12). Nous avons récemment ajouté des en-têtes de sécurité (CSP), assurez-vous d'utiliser la dernière version de `validator_app.html`.

---

## 5. Guide Technique : SDK Google Gen AI (v2.0+)

Ce projet utilise la nouvelle librairie unifiée `google-genai`. Voici les concepts clés pour étendre le code.

### Initialisation du Client

L'ancienne méthode `genai.configure()` est obsolète. Utilisez une instance de client :

```python
from google import genai
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
```

### Structuration des Requêtes (Contents)

Le SDK convertit automatiquement vos entrées en objets `types.Content`.

*   **Texte simple** : `contents='Pourquoi le ciel est bleu ?'`
*   **Multi-tours / Chat** :
    ```python
    from google.genai import types
    contents = [
        types.Content(role='user', parts=[types.Part.from_text(text='Bonjour')]),
        types.Content(role='model', parts=[types.Part.from_text(text='Salut !')])
    ]
    ```

### Configuration Avancée (System Instructions & Safety)

Utilisez `types.GenerateContentConfig` pour passer des instructions système ou régler la sécurité.

```python
config = types.GenerateContentConfig(
    system_instruction='Tu es un expert en diagrammes Mermaid.',
    temperature=0.5,
    safety_settings=[
        types.SafetySetting(
            category='HARM_CATEGORY_HATE_SPEECH',
            threshold='BLOCK_ONLY_HIGH',
        )
    ]
)
response = client.models.generate_content(model='gemini-2.0-flash', contents='...', config=config)
```

### Sortie Structurée (JSON Schema)

C'est la fonctionnalité clé utilisée dans ce projet pour garantir que l'IA génère du JSON valide.

```python
# Exemple avec un schéma JSON standard
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='Génère 3 questions sur les diagrammes de séquence.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema={
            "type": "OBJECT",
            "properties": {
                "questions": {"type": "ARRAY", "items": {"type": "STRING"}}
            }
        }
    ),
)
print(response.text) # Retourne un JSON string valide
```

### Support Pydantic

Vous pouvez aussi passer directement des modèles Pydantic (non utilisé actuellement dans le code mais supporté) :

```python
from pydantic import BaseModel

class DiagramRequest(BaseModel):
    title: str
    code: str

response = client.models.generate_content(
    ...,
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=DiagramRequest
    )
)
```

## 6. Gestion des Quotas et Modèles (Inférence)

Lors de l'exécution du script d'inférence (`gemini_fine_tune/perform_inference.py`), il est crucial de comprendre les limites de l'API Google Gemini (Free Tier) pour éviter les erreurs `429 RESOURCE_EXHAUSTED`.

### Limites de l'API (Free Tier)

Les quotas varient considérablement selon le modèle choisi (données observées en Janvier 2026). Voici un comparatif pour vous aider à choisir :

| Modèle | RPM (Req/min) | TPM (Tokens/min) | RPD (Req/jour) | Usage Recommandé |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma 3 (27b-it)** | **30** | 15,000 | **14,400** | **Validation de masse** (Recommandé) |
| **Gemini 3 Flash** | 5 | 250,000 | **20** | Tests unitaires / Preview |
| Gemini 2.5 Flash | 5 | 250,000 | 20 | Usage général (Quota faible) |
| Gemini 2.5 Flash-Lite | 10 | 250,000 | 20 | Tests rapides |

> **Note** : Les modèles "Flash" en Free Tier sont actuellement très limités en requêtes journalières (20 RPD). Privilégiez la famille **Gemma 3** pour traiter de gros volumes de données.

### Configuration du Script d'Inférence

Le script `perform_inference.py` a été amélioré pour gérer ces contraintes :

1.  **Menu de Sélection** : Au lancement, le script vous demande de choisir le **Base Model** (le modèle de référence).
    *   *Conseil* : Choisissez `gemma-3-27b-it` (ou un autre modèle Gemma) pour éviter les blocages de quota journalier si vous avez beaucoup de données.
2.  **Gestion du Fine-Tuned Model** :
    *   Par défaut, le script tente de se connecter à Vertex AI si l'ID commence par `projects/`.
    *   Pour utiliser un modèle standard ou un placeholder (pour tester le script sans modèle fine-tuné), modifiez votre fichier `.env` :
        ```dotenv
        # Commenter la config Vertex AI
        # FINE_TUNED_MODEL_ID=projects/...
        
        # Utiliser un modèle standard comme placeholder (ex: gemma-3-27b-it)
        # Cela permet de tester le script sans authentification Vertex AI complexe
        FINE_TUNED_MODEL_ID=gemma-3-27b-it
        ```
3.  **Délais Automatiques** :
    *   Le script détecte les erreurs 429 et attend automatiquement le temps demandé par l'API (souvent > 40s).
    *   Vous pouvez forcer un délai fixe via `.env` : `RATE_LIMIT_DELAY=4.0` (pour 15 RPM).

---

## 7. Génération Locale sur CPU (Qwen2.5-1.5B)

Cette section décrit comment utiliser le module `diagram_generation_on_cpu` pour entraîner et exécuter un modèle localement, sans dépendre de l'API Gemini. Nous utilisons **Qwen2.5-1.5B-Instruct**, un modèle léger capable de tourner sur CPU.

### Installation Spécifique

Ce module possède son propre environnement virtuel pour éviter les conflits avec le reste du projet.

```bash
cd diagram_generation_on_cpu

# 1. Créer l'environnement
python -m venv venv

# 2. Activer
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Installer les dépendances (Torch CPU, Transformers, PEFT)
pip install -r requirements.txt
```

### Test "Zero-Shot" (Sans entraînement)

Pour voir ce que le modèle sait faire de base, sans entraînement spécifique :

```bash
cd zero_shot_analysis
python test_zero_shot_mermaid.py
```
*Note : Le premier lancement téléchargera le modèle (~3GB). Le script affiche les diagrammes générés en temps réel.*

### Fine-Tuning (Entraînement)

Pour spécialiser le modèle sur vos données Mermaid (générées à l'étape 2 du tutoriel principal) :

```bash
# Revenir dans le dossier diagram_generation_on_cpu
cd .. 

# Lancer l'entraînement (ajustez les arguments selon votre RAM)
# --train_data_path : Chemin vers votre fichier JSONL d'entraînement
# --num_train_epochs : Nombre de passes (1 suffit souvent pour un test rapide)
# --output_dir : Dossier de sortie pour les checkpoints
python mermaid_finetune_cpu.py --train_data_path ../gemini_fine_tune/dataset/training_data.jsonl --num_train_epochs 1 --output_dir outputs/checkpoint-final
```

> **Note** : Si vous obtenez une erreur `ModuleNotFoundError: No module named 'torch'`, utilisez le script de lancement Bash :
> ```bash
> ./run_finetune.sh --train_data_path ../gemini_fine_tune/dataset/training_data.jsonl --num_train_epochs 1 --output_dir outputs/checkpoint-final
> ```

Les poids LoRA (l'adaptation du modèle) seront sauvegardés dans le dossier `outputs/`.

### Inférence (Utilisation du modèle entraîné)

Pour générer des diagrammes avec votre modèle affiné :

```bash
# Remplacez 'outputs/checkpoint-final' par le chemin réel de votre checkpoint
python diagram_inference.py --lora-path outputs/checkpoint-final --instruction "Create a sequence diagram for a login flow"
```

###  Exemple

 ./run_finetune.sh --train_data_path ../gemini_fine_tune/dataset/training_data.jsonl --num_train_epochs 1
============================================================
Starting CPU Fine-Tuning (Qwen2.5-1.5B)
============================================================
Using interpreter: venv/Scripts/python
============================================================
Initializing Fine-Tuning Script...
Loading heavy libraries (Torch, Transformers, PEFT)... this may take a moment on CPU.
============================================================
Libraries loaded successfully.
INFO:__main__:Loading model: Qwen/Qwen2.5-1.5B-Instruct
INFO:__main__:Model parameters: 1,543,714,304
INFO:__main__:Loading training data from ../gemini_fine_tune/dataset/training_data.jsonl
INFO:__main__:Loaded 1921 examples from ../gemini_fine_tune/dataset/training_data.jsonl
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1921/1921 [00:00<00:00, 290383.03 examples/s]
INFO:__main__:Loading eval data from ../gemini_fine_tune/dataset/validation_data.jsonl
INFO:__main__:Loaded 150 examples from ../gemini_fine_tune/dataset/validation_data.jsonl
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [00:00<00:00, 91886.32 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1921/1921 [00:01<00:00, 968.17 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [00:00<00:00, 940.83 examples/s]
INFO:__main__:Tokenized dataset. Example: {'input_ids': [38214, 374, 264, 1681, 311, 6923, 264, 13549, 304, 8755, 45242, 19482, 382, 14374, 6145, 510, 28468, 264, 2657, 42580, 13549, 369, 364, 15400, 705, 264, 501, 10554, 4427, 5443, 14158, 25, 364, 39253, 18626, 516, 364, 7688, 516, 323, 364, 16451, 4427, 758, 364, 39253, 18626, 1210, 364, 1806, 2011, 10554, 6, 320, 12338, 25, 220, 17, 11, 12089, 25, 2657, 701, 364, 14611, 5650, 336, 311, 10554, 98802, 5776, 6, 320, 12338, 25, 220, 19, 11, 19571, 25, 2657, 11, 5650, 336, 11, 10554, 701, 364, 14611, 17407, 311, 10554, 46425, 5776, 320, 15309, 21636, 320, 12338, 25, 220, 18, 11, 19571, 
....
....
....
============================================================
TRAINING ESTIMATION
============================================================
Examples: 1921
Epochs: 1.0
Batch Size (Effective): 4
Total Optimization Steps: 480
Estimated Time (CPU): ~8.0 hours (@ 60s/step)

⚠️  WARNING: Training will take a long time on CPU.
   Consider reducing dataset size or epochs for testing.
   You can use --max_steps 10 to run a quick test.
============================================================