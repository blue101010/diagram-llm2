
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Chargement des variables d'environnement depuis la racine du projet
# On suppose que le script est dans gemini_fine_tune/ et le .env à la racine
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(root_dir, '.env'))

# Chargement de la liste des modèles disponibles
models_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_limits_free.json')
try:
    with open(models_file_path, 'r') as f:
        models_data = json.load(f)
        # On prend par défaut "gemini-2.0-flash" s'il existe, sinon le premier de la liste
        default_model = "gemini-2.0-flash"
        if default_model in models_data:
            MODEL_ID = default_model
        elif models_data:
            MODEL_ID = list(models_data.keys())[0]
        else:
            MODEL_ID = "gemini-1.5-pro-001" # Fallback si le json est vide
except FileNotFoundError:
    print(f"Attention: {models_file_path} non trouvé. Utilisation du modèle par défaut.")
    MODEL_ID = "gemini-1.5-pro-001"

# Configuration
PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("VERTEX_PROJECT_ID")
LOCATION = os.getenv("LOCATION") or os.getenv("VERTEX_LOCATION", "us-central1")

if not PROJECT_ID:
    print("Attention: PROJECT_ID (ou VERTEX_PROJECT_ID) non trouvé dans le fichier .env. Veuillez le configurer.")
    PROJECT_ID = "votre-projet-id" # Valeur par défaut pour éviter le crash immédiat, mais l'appel échouera sans auth

def predict_vertex(prompt_text):
    """
    Effectue une inférence en utilisant le SDK Google Gen AI unifié (compatible Vertex AI).
    """
    # Initialisation du client Google Gen AI avec le mode Vertex AI activé
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )

    # Génération de contenu
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            max_output_tokens=2048,
            temperature=0.9,
            top_p=1
        )
    )
    
    return response.text

if __name__ == "__main__":
    # Exemple d'utilisation
    user_prompt = "Expliquez le concept de fine-tuning dans le contexte des LLMs."
    
    try:
        print(f"Prompt: {user_prompt}")
        print("-" * 20)
        response = predict_vertex(user_prompt)
        print("Réponse du modèle (Vertex AI via Google Gen AI SDK):")
        print(response)
    except Exception as e:
        error_str = str(e)
        if "BILLING_DISABLED" in error_str or "requires billing to be enabled" in error_str:
            print("\nERREUR CRITIQUE : La facturation (Billing) n'est pas activée pour ce projet Google Cloud.")
            print(f"Veuillez activer la facturation pour le projet '{PROJECT_ID}' ici :")
            print(f"https://console.developers.google.com/billing/enable?project={PROJECT_ID}")
            print("Note : Vertex AI est un service payant (ou nécessite un compte de facturation actif même pour le tiers gratuit).")
        else:
            print(f"Une erreur s'est produite : {e}")
            print("Assurez-vous d'avoir configuré vos identifiants Google Cloud (gcloud auth application-default login) et installé le package : pip install google-genai")
