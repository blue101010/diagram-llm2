# Script pour installer le Google Cloud SDK sur Windows

Write-Host "Vérification de l'installation de Google Cloud SDK..."

if (Get-Command gcloud -ErrorAction SilentlyContinue) {
    Write-Host "Google Cloud SDK est déjà installé."
    gcloud --version
    exit
}

Write-Host "Google Cloud SDK n'est pas trouvé. Téléchargement de l'installateur..."

$installerUrl = "https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe"
$installerPath = "$env:TEMP\GoogleCloudSDKInstaller.exe"

try {
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
    Write-Host "Téléchargement terminé."
}
catch {
    Write-Error "Erreur lors du téléchargement de l'installateur : $_"
    exit 1
}

Write-Host "Lancement de l'installation..."
Write-Host "Une fenêtre d'installation va s'ouvrir. Veuillez suivre les instructions à l'écran."
Write-Host "IMPORTANT : Assurez-vous de cocher les options pour ajouter gcloud au PATH système."

Start-Process -FilePath $installerPath -Wait

Write-Host "Installation terminée."
Write-Host "Veuillez REDÉMARRER VS Code (fermer et rouvrir) pour que la commande 'gcloud' soit reconnue."
