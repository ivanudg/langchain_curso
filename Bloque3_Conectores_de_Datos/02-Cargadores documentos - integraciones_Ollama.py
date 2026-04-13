## Importar librerías e instancia de modelo de chat
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

#  Integración Wikipedia
from langchain_community.document_loaders import WikipediaLoader # pip install wikipedia en una terminal

from langchain_ollama import ChatOllama

# Generar instancia de modelo de chat
chat = ChatOllama(
    model='gpt-oss:20b'
)

def responder_wikipedia(persona, pregunta_arg):
    # Obtener artículo de wikipedia
    docs = WikipediaLoader(query=persona, lang='es', load_max_docs=10) #parámetros posibles en: https://python.langchain.com/v0.2/docs/integrations/document_loaders/wikipedia/
    contexto_extra = docs.load()[0].page_content #para que sea más rápido solo pásamos el primer documento [0] como contexto extra

    # Pregunta de usuario
    human_prompt = HumanMessagePromptTemplate.from_template('Responde a esta pregunta\n{pregunta}, aquí tienes contenido extra:\n{contenido}')

    # Construir prompt
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

    # Resultado
    result = chat.invoke(
        chat_prompt.format_prompt(pregunta=pregunta_arg, contenido=contexto_extra).to_messages()
    )

    print("*"*150)
    print("RESPUESTA A PREGUNTA CON CONTEXTO DE WIKIPEDIA")
    print("*"*150+'\n')
    print(result.content)

# Ejemplo de uso
responder_wikipedia("Fernando Alonso","¿Cuándo nació?")