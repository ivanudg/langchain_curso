from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Nueva forma de inicializar el cliente
client = genai.Client(api_key=api_key)


try:
    # En la nueva librería, el método es models.generate
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents="Explain how AI works in a few words"
    )
    print("Respuesta de la nueva API:")
    print(response.text)
except Exception as e:
    print(f"Error con la nueva librería: {e}")