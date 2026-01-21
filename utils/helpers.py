import os
from datetime import datetime

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def download_model_instructions():
    return """
To download LLaMA model:
1. Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
2. Download: llama-2-7b-chat.Q4_K_M.gguf
3. Place in: jarvis-ai-assistant/models/
4. Or use smaller model for testing
    """