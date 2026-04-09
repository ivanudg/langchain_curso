## Importar librerías e instancia de modelo de chat
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

## Obtener llave de API de OpenAI y generar instancia de modelo de chat
import dotenv
dotenv.load_dotenv()
import os
api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(openai_api_key=api_key)

## Parsear una lista de elementos separados por coma
from langchain_core.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions() #Nos devuelve las instrucciones que va a pasar al LLM en función del parseador concreto
print(format_instructions)

#Respuesta imaginaria
respuesta = "coche, árbol, carretera"
print(output_parser.parse(respuesta)) #Nos devuelve la respuesta parseada, es decir, una lista de elementos separados por coma

#Creamos la plantilla de usuario (human_template) con la concatenación de la variable "request" (la solicitud) y la variable "format_instructions" 
# con las instrucciones adicionales que le pasaremos al LLM
human_template = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

#Creamos el prompt y le damos formato a las variables
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
resp = chat_prompt.format_prompt(
    request="dime 5 características de los coches americanos",
    format_instructions = format_instructions #Las instrucciones son las que proporciona el propio parseador
)
print(resp.to_messages()) #Nos devuelve el mensaje formateado con la solicitud y las instrucciones  

#Transformamos el objeto prompt a una lista de mensajes y lo guardamos en "solicitud_completa" que es lo que pasaremos al LLM finalmente
solicitud_completa = resp.to_messages()

result = chat.invoke(solicitud_completa) #Pasamos la solicitud completa al LLM
salida = output_parser.parse(result.content) #Parseamos la salida del LLM con el parseador que hemos creado
print(f'Salida: {salida}') #Nos devuelve la salida parseada, es decir, una lista de características de los coches americanos