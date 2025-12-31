# Tutoriel et Plan de Test : Projet Diagram-LLM2

Ce document sert de guide complet pour comprendre, installer et tester le projet **Diagram-LLM2**. Ce projet vise à créer des jeux de données synthétiques de haute qualité pour entraîner des modèles d'IA (comme Gemini ou Phi-4) à générer des diagrammes **Mermaid.js**.

---

## 1. Introduction

Le but de ce projet est d'automatiser la création de données d'entraînement pour les diagrammes. Au lieu d'écrire manuellement des milliers d'exemples, nous utilisons un LLM puissant (Gemini Pro) pour générer des exemples variés basés sur la documentation officielle de Mermaid.

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

1.  **Python 3.10** ou supérieur installé.
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
2.  Ouvrez `previewer/index.html` dans votre navigateur.
3.  Vous pouvez maintenant faire défiler les cartes montrant le "Prompt" (la demande utilisateur) et le "Diagramme" (le résultat visuel).

---

## Résolution de problèmes courants

*   **Erreur API Key** : Vérifiez que la variable d'environnement `GEMINI_API_KEY` est bien définie dans votre terminal ou votre fichier `.env`.
*   **Erreur CSP dans le navigateur** : Si le validateur ne charge pas Mermaid, vérifiez la console du navigateur (F12). Nous avons récemment ajouté des en-têtes de sécurité (CSP), assurez-vous d'utiliser la dernière version de `validator_app.html`.
