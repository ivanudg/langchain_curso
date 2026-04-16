#  ChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Crear una instancia del modelo de lenguaje
chat = ChatOllama(model="gpt-oss:20b-cloud")

#Definimos el objeto de histórico de mensajes
from langchain_classic.memory import ChatMessageHistory

history = ChatMessageHistory()
consulta = "Hola, ¿cómo estás? Necesito ayudar para reconfigurar el router"

#Vamos guardando en el objeto "history" los mensajes de usuario y los mensajes AI que queramos
history.add_user_message(consulta)
resultado = chat.invoke([HumanMessage(content=consulta)])
history.add_ai_message(resultado.content)

print("Historial de mensajes:")
print(history.messages)