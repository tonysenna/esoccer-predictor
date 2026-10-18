@echo off
title Compartir e-Soccer Predictor con Socio
color 0A

cd /d "%~dp0"

:: Verificar si el servidor Streamlit esta corriendo
netstat -ano | findstr :8501 | findstr LISTENING >nul
if %errorlevel% neq 0 (
    echo Iniciando el servidor local primero...
    start /min "" py -3.12 -m streamlit run src/ui/app_streamlit.py --server.port 8501 --server.headless true
    timeout /t 3 >nul
)

py -3.12 share_tunnel.py

pause
