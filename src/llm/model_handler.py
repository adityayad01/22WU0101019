from llama_cpp import Llama
import os

class LLaMAHandler:
    def __init__(self, model_path=None):
        self.model_path = model_path or "models/llama-2-7b-chat.gguf"
        self.llm = None
        
    def load_model(self):
        try:
            print(f"Loading model from {self.model_path}...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=0
            )
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_response(self, prompt, max_tokens=512, temperature=0.7):
        if not self.llm:
            return "Model not loaded."
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["User:", "\n\n"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error: {e}"
    
    def create_chat_prompt(self, user_message, context=""):
        prompt = f"""You are Jarvis, a helpful AI assistant.

Context: {context}

User: {user_message}
Assistant:"""
        return prompt