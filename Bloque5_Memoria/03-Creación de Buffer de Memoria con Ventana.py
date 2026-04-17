## Importar librerías iniciales e instancia de modelo de chat
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_ollama import ChatOllama

## Instancia de modelo de chat
llm = ChatOllama(model='gpt-oss:20b-cloud')

#  Crear objeto ConversationBufferWindowMemory
#k indica el número de iteraciones (pareja de mensajes human-AI) que guardar
memory = ConversationBufferWindowMemory(k=1)

## Conectar una conversación a la memoria
#Creamos una instancia de la cadena conversacional con el LLM y el objeto de memoria
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
#Ejemplo con RunnableWithMessageHistory: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html
conversation.predict(input="Hola, ¿cómo estás?") # Primera iteración
conversation.predict(input="Necesito un consejo para tener un gran día") # Segunda iteración

print("\nContenido de la memoria:\n")
print(memory.buffer) #k limita el número de interacciones