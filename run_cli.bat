@echo off
title e-Soccer Predictor CLI
color 0B
echo ===============================================================================
echo                e-Soccer FIFA Predictive Engine (CLI Terminal)
echo ===============================================================================
echo.

cd /d "%~dp0"
py -3.12 main.py --help
echo.
echo Ejemplo de prediccion con cuotas:
echo   py -3.12 main.py predict "Lio" "Danny" --format 5x2 --odds-1x2 2.20 4.10 2.80 --odds-o25 1.55 2.35 --odds-btts 1.45 2.55
echo.
echo Ejemplo de backtesting:
echo   py -3.12 main.py backtest --format 5x2 --bankroll 1000 --min-edge 0.03
echo.
cmd /k
