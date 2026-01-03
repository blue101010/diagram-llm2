# Tutoriel : Fine-Tuning de Qwen2.5-1.5B sur CPU pour la Génération de Diagrammes

Ce tutoriel explique comment adapter le projet `diagram-llm2` pour utiliser un modèle local **Qwen2.5-1.5B** fine-tuné sur CPU, en remplacement des API Cloud (Gemini/OpenAI).

Cette approche est inspirée par l'article [Fine-tuning small language models on a basic desktop PC](https://blog.bjdean.id.au/2025/06/fine-tuning-small-language-models-on-a-basic-desktop-pc/).

## Faisabilité sur CPU

Bien que l'article de référence utilise un modèle de 0.5B paramètres, nous utilisons ici **Qwen2.5-1.5B-Instruct**.

*   **Faisabilité** : Qwen2.5-1.5B est suffisamment petit pour que le fine-tuning via **PEFT/LoRA** soit possible sur un CPU moderne multicœur.
*   **RAM requise** : 32 Go à 64 Go de RAM sont recommandés.
*   **Temps d'entraînement** : Le passage de 0.5B à 1.5B multiplie le temps de calcul par environ 3. Attendez-vous à des durées allant de quelques dizaines de minutes à quelques heures par expérience, selon la longueur des séquences et la taille du batch.

## Architecture et Scripts

Nous avons ajouté un dossier `diagram_generation_on_cpu/` contenant les outils nécessaires pour remplacer le backend cloud :

1.  **`mermaid_finetune_cpu.py`** : Script d'entraînement optimisé pour CPU. Il détecte et charge automatiquement vos données existantes (format Gemini JSONL) et entraîne un adaptateur LoRA.
2.  **`diagram_inference.py`** : Script d'inférence capable de charger le modèle de base seul (Zero-Shot) ou avec l'adaptateur LoRA fine-tuné pour générer des diagrammes Mermaid.

## Guide Étape par Étape

### 1. Installation

Il est recommandé d'utiliser un environnement virtuel pour isoler les dépendances. Voici les commandes optimisées pour **Git Bash** (Windows) :

```bash
cd diagram_generation_on_cpu

# 1. Créer l'environnement virtuel (nommé 'venv')
python -m venv venv

# 2. Activer l'environnement
source venv/Scripts/activate

# 3. Mettre à jour pip et installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Test Zero-Shot (Avant Fine-Tuning)

Avant d'entraîner, vérifiez comment le modèle de base **Qwen2.5-1.5B-Instruct** se comporte sur vos données de validation existantes. Cela permet de voir si le modèle "comprend" déjà le Mermaid sans entraînement.

```bash
# Test sur 3 exemples du fichier de validation existant
python diagram_inference.py --limit 3
```

*Le script téléchargera automatiquement le modèle (~3 Go) lors de la première exécution.*

### 3. Lancement du Fine-Tuning

Lancez l'entraînement en utilisant votre fichier `training_data.jsonl` existant (celui généré pour Gemini). Le script est configuré pour utiliser des paramètres conservateurs adaptés au CPU (batch size 1, accumulation de gradients).

```bash
python mermaid_finetune_cpu.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_data_path ../gemini_fine_tune/dataset/training_data.jsonl \
    --output_dir ./lora_mermaid_output \
    --num_train_epochs 3
```

> **Astuce (Bash)** : Si vous rencontrez l'erreur `ModuleNotFoundError: No module named 'torch'`, utilisez le script de lancement :
> ```bash
> ./run_finetune.sh --train_data_path ../gemini_fine_tune/dataset/training_data.jsonl --num_train_epochs 1
> ```

### 4. Inférence avec le Modèle Fine-Tuné

Une fois l'entraînement terminé, testez votre nouvel adaptateur LoRA pour voir les améliorations :

```bash
python diagram_inference.py \
    --lora-path ./lora_mermaid_output \
    --limit 5
```

### 5. Intégration (Remplacement du Backend)

Pour utiliser ce modèle local à la place de Gemini dans l'application principale, vous pouvez importer la classe `DiagramGenerator` depuis `diagram_inference.py`.

Exemple d'intégration Python :

```python
from diagram_generation_on_cpu.diagram_inference import DiagramGenerator

# Initialisation (lent au démarrage, charge le modèle en RAM)
generator = DiagramGenerator(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    lora_adapter_path="./diagram_generation_on_cpu/lora_mermaid_output",
    device="cpu"
)

# Génération
instruction = "Create a flowchart for a login process"
result = generator.generate_diagram(instruction)
print(result["mermaid"])
```

---
*Adapté de diagram-llm2 pour une exécution locale et privée.*
