#!/bin/bash

# Tentative de détection automatique du chemin d'installation de Google Cloud SDK
# pour l'ajouter au PATH de Bash.

echo "Recherche de gcloud..."

# Liste des chemins potentiels standards sous Windows
POSSIBLE_PATHS=(
    "$HOME/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin"
    "/c/Users/$USERNAME/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin"
    "/c/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin"
    "/c/Program Files/Google/Cloud SDK/google-cloud-sdk/bin"
)

FOUND_PATH=""

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path/gcloud" ] || [ -f "$path/gcloud.cmd" ]; then
        FOUND_PATH="$path"
        break
    fi
done

if [ -n "$FOUND_PATH" ]; then
    echo "✅ gcloud trouvé dans : $FOUND_PATH"
    
    # Ajouter au PATH pour la session courante
    export PATH="$PATH:$FOUND_PATH"
    
    # Ajouter au .bashrc pour la persistance
    BASH_RC="$HOME/.bashrc"
    
    # Créer .bashrc s'il n'existe pas
    if [ ! -f "$BASH_RC" ]; then
        touch "$BASH_RC"
    fi

    if ! grep -q "$FOUND_PATH" "$BASH_RC"; then
        echo "" >> "$BASH_RC"
        echo "# Google Cloud SDK" >> "$BASH_RC"
        echo "export PATH=\"\$PATH:$FOUND_PATH\"" >> "$BASH_RC"
        echo "✅ Chemin ajouté à $BASH_RC"
    else
        echo "ℹ️ Le chemin est déjà présent dans $BASH_RC"
    fi
    
    echo ""
    echo "Test de la commande :"
    gcloud --version
    
    echo ""
    echo "🎉 Configuration terminée !"
    echo "Pour appliquer les changements à votre terminal actuel, lancez :"
    echo "source ~/.bashrc"
else
    echo "❌ Impossible de trouver le Google Cloud SDK dans les emplacements par défaut."
    echo "Si vous l'avez installé dans un dossier personnalisé, vous devez ajouter le dossier 'bin' à votre PATH manuellement."
fi
