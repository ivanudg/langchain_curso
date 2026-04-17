## Importar librerías iniciales e instancia de modelo de chat
from langchain_classic import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama

## Crear instancia de modelo de chat
llm = ChatOllama(model="gpt-oss:20b-cloud")

#  Crear objeto ConversationBufferMemory
memory = ConversationBufferMemory()

## Conectar una conversación a la memoria
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

#Ejemplo con RunnableWithMessageHistory: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html

#Lanzamos el primer prompt (human message)
conversation.predict(input="Hola, necesito saber cómo usar mis datos históricos para crear un bot de preguntas y respuestas")

#Lanzamos el segundo prompt (human message)
conversation.predict(input="Necesito más detalle de cómo implementarlo")

print(memory.buffer)
print("*"*300)

#Cargamos la variable de memoria
memory.load_memory_variables({})

## Guardar y Cargar la memoria (posterior uso)
conversation.memory

import pickle
from pathlib import Path

#Crea un objeto binario con todo el objeto de la memoria
pickled_str = pickle.dumps(conversation.memory)
path_memory = Path( Path.cwd(), 'Bloque5_Memoria', 'Memory')
File_memory = Path(path_memory, "memory.pkl")

print(f'La memoria se guardará en: {File_memory}')

#wb para indicar que escriba un objeto binario, en este caso en la misma ruta que el script
with open(str(File_memory), "wb") as f:
    f.write(pickled_str)

memoria_caragda = open(str(File_memory), "rb").read()
#Creamos una nueva instancia de LLM para asegurar que está totalmente limpia
llm= ChatOllama(model="gpt-oss:20b-cloud")
conversation_cargada = ConversationChain(
    llm=llm, 
    memory=pickle.loads(memoria_caragda), 
    verbose=True
)

print(conversation_cargada.memory.buffer)

