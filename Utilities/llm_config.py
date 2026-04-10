## Obtener llave de API de OpenAI y generar instancia de modelo de chat
import dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def get_chat()->ChatOpenAI:
    global api_key
    return ChatOpenAI(openai_api_key=api_key)

def get_embeddings()->OpenAIEmbeddings:
    global api_key
    return OpenAIEmbeddings(openai_api_key=api_key)