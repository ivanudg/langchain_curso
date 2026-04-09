## Obtener llave de API de OpenAI y generar instancia de modelo de chat
import dotenv
import os
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

def get_chat()->ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(openai_api_key=api_key)