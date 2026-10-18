import os
import sys

# Asegurar que la raíz del proyecto está en PYTHONPATH para importaciones src.*
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Importar y ejecutar la interfaz de Streamlit
import src.ui.app_streamlit
