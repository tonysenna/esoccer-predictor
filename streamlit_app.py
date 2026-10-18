# e-Soccer Predictor & Value Bet Engine - v1.2 (Gamers & Teams Selectbox)
import os
import sys

# Asegurar que la raíz del proyecto está en PYTHONPATH para importaciones src.*
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import runpy

# Limpiar cache de módulos internos 'src.*' para forzar la recarga del código actualizado
for mod in list(sys.modules.keys()):
    if mod.startswith("src."):
        del sys.modules[mod]

app_path = os.path.join(ROOT_DIR, "src", "ui", "app_streamlit.py")
runpy.run_path(app_path, run_name="__main__")
