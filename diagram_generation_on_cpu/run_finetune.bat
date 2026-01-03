@echo off
REM Script pour lancer le fine-tuning avec l'environnement virtuel correct
REM Utilisation: run_finetune.bat [arguments]

echo ============================================================
echo Lancement du Fine-Tuning sur CPU (Qwen2.5-1.5B)
echo ============================================================

REM Vérifier si le venv existe
if not exist "venv\Scripts\python.exe" (
    echo ERREUR: L'environnement virtuel n'est pas trouve dans 'venv'.
    echo Veuillez executer: python -m venv venv && pip install -r requirements.txt
    pause
    exit /b 1
)

REM Lancer le script avec le python du venv
echo Utilisation de l'interpreteur: venv\Scripts\python.exe
venv\Scripts\python.exe mermaid_finetune_cpu.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Une erreur est survenue lors de l'execution.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Fin du traitement.
pause
