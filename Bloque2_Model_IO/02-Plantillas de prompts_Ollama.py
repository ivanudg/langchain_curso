## Importar librerías de templates
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_ollama import ChatOllama

## Generar plantillas de prompts
#Creamos la plantilla del sistema (system_template)
system_template = "Eres una IA especializada en coches de tipo {tipo_coches} y generar artículos que se leen en {tiempo_lectura}."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
print(f'system_message_prompt Variables: {system_message_prompt.input_variables}')

#Creamos la plantilla de usuario (human_template)
human_template = "Necesito un artículo para vehículos con motor {peticion_tipo_motor}"
human_message_prompt = SystemMessagePromptTemplate.from_template(human_template)
print(f'human_message_prompt Variables: {human_message_prompt.input_variables}')

#Creamos una plantilla de chat con la concatenación tanto de mensajes del sistema como del humano
chat_promt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
print(f'Chat Promt Variables: {chat_promt.input_variables}')

# Completar el chat gracias al formateo de los mensajes
chat_promt.format_prompt(peticion_tipo_motor='Hibrido enchufable', tiempo_lectura='10 min', tipo_coches='Japoneses')

#Transformamos el objeto prompt a una lista de mensajes y lo guardamos en "solicitud_completa" que es lo que pasaremos al LLM finalmente
solicitud_completa = chat_promt.format_prompt(peticion_tipo_motor='Hibrido enchufable', tiempo_lectura='10 min', tipo_coches='Japoneses').to_messages()
print(f'Solicitud completa: {solicitud_completa}')

## Obtener el resultado de la respuesta formateada
try:
    chat = ChatOllama(
        model='gpt-oss:20b'
    )
    resultado = chat.invoke(solicitud_completa)
    print(f'Resultado: {resultado}')
    print(f'Resultado content: {resultado.content}')
except Exception as e:
    print(f'Error al obtener el resultado: {e}')
