import os
import sys
import subprocess
import platform

# Identify the correct python and streamlit paths
VENV_DIR = ".venv"
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    PYTHON_BIN = os.path.join(VENV_DIR, "Scripts", "python.exe")
    STREAMLIT_BIN = os.path.join(VENV_DIR, "Scripts", "streamlit.exe")
else:
    PYTHON_BIN = os.path.join(VENV_DIR, "bin", "python")
    STREAMLIT_BIN = os.path.join(VENV_DIR, "bin", "streamlit")

def run_command(command_list):
    try:
        subprocess.run(command_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Executable not found. Did you create the virtual environment in '{VENV_DIR}'?")
        sys.exit(1)

def ingest():
    print("--- Running Ingestion ---")
    run_command([PYTHON_BIN, "day2_rag/ingest.py"])

def run_app():
    print("--- Starting Streamlit App ---")
    run_command([STREAMLIT_BIN, "run", "day3_agents/app.py", "--server.headless", "true"])

def setup():
    print("--- Setting up environment ---")
    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("Using 'uv' for fast installation...")
        run_command(["uv", "pip", "install", "-r", "requirements.txt"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("'uv' not found. Falling back to standard 'pip'...")
        run_command([PYTHON_BIN, "-m", "pip", "install", "-r", "requirements.txt"])

def show_help():
    print("DocuAgent Management Script")
    print("Usage:")
    print("  python manage.py ingest  - Run the data ingestion pipeline")
    print("  python manage.py run     - Start the Streamlit application")
    print("  python manage.py setup   - Install dependencies from requirements.txt")
    print("  python manage.py help    - Show this help message")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "ingest":
        ingest()
    elif command == "run":
        run_app()
    elif command == "setup":
        setup()
    elif command in ["help", "--help", "-h"]:
        show_help()
    else:
        print(f"Unknown command: {command}")
        show_help()
