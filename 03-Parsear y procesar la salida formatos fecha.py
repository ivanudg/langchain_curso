## Parsear formatos de fecha
# Importa el tipo datetime para poder trabajar con fechas/horas como objetos Python (no solo texto)
from datetime import datetime

# Importa BaseModel y Field de Pydantic para definir un "esquema" de datos validable (JSON -> objeto tipado)
from pydantic import BaseModel, Field

# Importa el parser de LangChain que convierte la salida del LLM en un objeto Pydantic validado
from langchain_core.output_parsers import PydanticOutputParser

# Importa la plantilla de prompt para construir prompts de chat (mensajes tipo system/human/ai)
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Importa el wrapper del modelo de chat de OpenAI para usarlo dentro de LangChain
from langchain_openai import ChatOpenAI

## Obtener llave de API de OpenAI y generar instancia de modelo de chat
import dotenv
import os

dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=api_key)

# Define un modelo Pydantic que representa exactamente el formato de salida que queremos del LLM
class FechaRespuesta(BaseModel):
    # Define el campo 'fecha' como datetime y añade una descripción para guiar al LLM
    fecha: datetime = Field(description="Fecha en formato ISO 8601 (por ejemplo: 1776-07-04T00:00:00Z)")

# Crea el parser que forzará el formato de salida del LLM cumpla el esquema 'FechaRespuesta'
output_parser = PydanticOutputParser(pydantic_object=FechaRespuesta)
template_text = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(template_text)

# Crea el prompt de chat a partir de una lista de mensajes (en este caso solo uno humano)
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
chat_format = chat_prompt.format_prompt(
    request="¿Cuándo es el día de la declaración de independencia de los EEUU?",
    format_instructions=output_parser.get_format_instructions() # Añade las instrucciones de formato del parser al prompt   
)
#print(f'chat_format: {chat_format}') # Imprime el prompt formateado con la solicitud y las instrucciones de formato
solicitud_completa = chat_format.to_messages() # Convierte el prompt a una lista de mensajes para pasarlo al LLM
result = chat.invoke(solicitud_completa) # Pasa la solicitud completa al LLM y guarda la respuesta
salida = output_parser.parse(result.content) # Parsea la salida del LLM usando el
print(f'Salida: {salida}') # Imprime la salida parseada, que es un objeto FechaRespuesta con el campo 'fecha' como datetime