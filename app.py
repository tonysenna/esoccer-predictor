#!/usr/bin/env python3
import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = [sys.executable, os.path.join(PROJECT_DIR, "main.py")] + sys.argv[1:]
        subprocess.run(cmd, cwd=PROJECT_DIR)
    else:
        print("Iniciando interfaz web Streamlit...")
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            os.path.join(PROJECT_DIR, "src", "ui", "app_streamlit.py"),
            "--server.port",
            "8501",
            "--server.headless",
            "false",
        ]
        subprocess.run(cmd, cwd=PROJECT_DIR)
