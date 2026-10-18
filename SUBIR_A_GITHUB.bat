@echo off
setlocal
set "PATH=C:\Program Files\Git\cmd;%PATH%"
cd /d "%~dp0"

echo ===================================================================
echo Subiendo e-Soccer Predictor a GitHub (tonysenna/esoccer-predictor)...
echo ===================================================================
echo.
echo Si te pide iniciar sesion, se abrira una ventana o el navegador.
echo Pulsa en 'Sign in with your browser' / 'Authorize'.
echo.

"C:\Program Files\Git\cmd\git.exe" push -u origin main

echo.
echo ===================================================================
echo Proceso terminado. Ya puedes cerrar esta ventana.
echo ===================================================================
pause
