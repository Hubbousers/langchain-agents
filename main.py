import os

from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
nvidia_key_1 = os.getenv("NVIDIA_API_KEY")
nvidia_key_2 = os.getenv("NVIDIA_API_KEY_2")
mistral_key_1 = os.getenv("MISTRAL_API_KEY")
mistral_key_2 = os.getenv("MISTRAL_API_KEY_2")
mistral_key_3 = os.getenv("MISTRAL_API_KEY_3")

print("Google Gemma API Key loaded:", google_api_key is not None)
print("NVIDIA API Key 1 loaded:", nvidia_key_1 is not None)
print("NVIDIA API Key 2 loaded:", nvidia_key_2 is not None)
print("Mistral API Key 1 loaded:", mistral_key_1 is not None)
print("Mistral API Key 2 loaded:", mistral_key_2 is not None)
print("Mistral API Key 3 loaded:", mistral_key_3 is not None)
print("Hello from LangChain course!")
