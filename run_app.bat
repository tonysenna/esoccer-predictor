@echo off
title e-Soccer FIFA Predictive Engine - Interfaz Grafica
color 0A
echo ===============================================================================
echo                e-Soccer FIFA Predictive Engine y Value Bets
echo      GT Leagues y Esports Battle (4x2, 5x2, 6x2) - Dixon-Coles y ELO
echo ===============================================================================
echo.

cd /d "%~dp0"

:: Verificar si el servidor ya esta corriendo en el puerto 8501
netstat -ano | findstr :8501 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo El servidor ya esta activo en el puerto 8501.
    echo Abriendo la aplicacion en tu navegador...
    start http://localhost:8501
    goto :end
)

echo Iniciando el servidor y cargando modelos...
echo Tu navegador se abrira automaticamente en http://localhost:8501
echo.

py -3.12 -m streamlit run src/ui/app_streamlit.py --server.port 8501 --server.headless false

:end
pause
