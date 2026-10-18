import subprocess
import re
import sys
import time
import os

print("=" * 75)
print("     INICIANDO GENERADOR DE ENLACE PUBLICO (CLOUDFLARE HTTPS)")
print("=" * 75)
print("\n[1/2] Conectando con los servidores seguros de Cloudflare...")

# Iniciar cloudflared como subproceso
proc = subprocess.Popen(
    ["cloudflared.exe", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    encoding="utf-8",
    errors="replace"
)

tunnel_url = None
start_time = time.time()

while True:
    line = proc.stdout.readline()
    if not line and proc.poll() is not None:
        break
    
    # Buscar URL trycloudflare.com
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        tunnel_url = match.group(0)
        break

    if time.time() - start_time > 20:
        break

if tunnel_url:
    # Intentar copiar al portapapeles de Windows automaticamente
    try:
        import ctypes
        subprocess.run(f"echo {tunnel_url}| clip", shell=True)
        clipboard_msg = "(¡ENLACE COPIADO AUTOMATICAMENTE AL PORTAPAPELES!)"
    except Exception:
        clipboard_msg = ""

    print("\n" + "=" * 75)
    print("  LISTO! TU ENLACE PUBLICO Y SEGURO (HTTPS) ES:")
    print(f"  -->  {tunnel_url}  <--")
    print(f"  {clipboard_msg}")
    print("=" * 75)
    print("\nINSTRUCCIONES PARA TU SOCIO:")
    print("1. Dale 'Pegar' (Ctrl + V) en el chat de WhatsApp con tu socio.")
    print("2. Tu socio entrara directamente desde su movil o PC.")
    print("\n>>> DEJA ESTA VENTANA ABIERTA MIENTRAS TU SOCIO ESTE USANDO LA WEB <<<\n")
    
    # Mantener el proceso vivo
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nCerrando tunel...")
        proc.terminate()
else:
    print("\n[!] No se pudo capturar el enlace automaticamente. Mostrando logs:")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
