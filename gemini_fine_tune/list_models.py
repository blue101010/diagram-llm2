import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

def list_models():
    client = genai.Client(api_key=API_KEY)
    try:
        # The SDK might have a different way to list models, but let's try the standard way if possible
        # or just try to generate content with a known model to check connection
        # The error message suggested "Call ListModels".
        # In the new SDK, it might be client.models.list()
        
        print("Attempting to list models...")
        # Note: The specific method to list models depends on the SDK version.
        # Assuming google-genai SDK.
        
        # Try v1beta
        print("\n--- Models (v1beta) ---")
        try:
            # This is a guess at the SDK method, if it fails I'll try another way
            # Inspecting the client object might be hard in a script without docs, 
            # but usually it's client.models.list()
            for m in client.models.list():
                print(f"Name: {m.name}")
                print(f"  DisplayName: {m.display_name}")
                # print(f"  SupportedGenerationMethods: {m.supported_generation_methods}")
                # Print available attributes to debug
                # print(f"  Attributes: {dir(m)}")

        except Exception as e:
            print(f"Error listing models: {e}")

    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    print("Starting list_models.py...")
    list_models()
