import sys
sys.path.append('src')

from chatbot.interface import ChatbotInterface
from utils.helpers import ensure_directory, download_model_instructions

def main():
    print("=" * 50)
    print("Starting Jarvis AI Assistant")
    print("=" * 50)
    
    # Ensure models directory exists
    ensure_directory("models")
    
    print(download_model_instructions())
    
    # Run the chatbot interface
    chatbot = ChatbotInterface()
    chatbot.run()

if __name__ == "__main__":
    main()
