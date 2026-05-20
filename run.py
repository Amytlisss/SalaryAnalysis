import subprocess
import sys
import os
import webbrowser
import time
import threading
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def main():
    port = 8000
    
    print("[INFO] Запуск SalaryAnalysis")
    print("[INFO] Установка зависимостей...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pandas", "scipy", "statsmodels", "matplotlib", "seaborn", "openpyxl", "fastapi", "uvicorn", "python-multipart"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка установки: {e}")
        print("[INFO] Попробуйте установить зависимости вручную:")
        print("pip install numpy pandas scipy statsmodels matplotlib seaborn openpyxl fastapi uvicorn python-multipart")
        return
    
    print("[INFO] Запуск сервера FastAPI...")
    
    def run_server():
        os.chdir("backend")
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)])
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print("[INFO] Ожидание запуска сервера...")
    time.sleep(3)
    
    ui_url = f"http://127.0.0.1:{port}/ui"
    
    print("СЕРВЕР ЗАПУЩЕН")
    print(f"Веб-интерфейс: {ui_url}")
    print(f"API документация: http://127.0.0.1:{port}/docs")
    print("Для остановки сервера нажмите Ctrl+C")
    
    webbrowser.open(ui_url)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Остановка приложения")

if __name__ == "__main__":
    os.makedirs("backend", exist_ok=True)
    os.makedirs("frontend", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    main()