from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_ollama import ChatOllama

# Crear una instancia del modelo de lenguaje
llm = ChatOllama(model='gpt-oss:20b-cloud')

#  Crear objeto ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# Creamos un prompt cuya respuesta hará que se sobrepase el límite de tokens y por tanto sea recomendable resumir la memoria
plan_viaje = '''Este fin de semana me voy de vacaciones a la playa, estaba pensando algo que fuera bastante relajado, pero necesito 
un plan detallado por días con qué hacer en familia, extiéndete todo lo que puedas'''

## Creamos una nueva conversación con un buffer de memoria resumida
memory = ConversationSummaryMemory(llm=llm, max_token_limit=100)
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

conversation.predict(input=plan_viaje)

print("\nVarieables de memorya: \n")
print(memory.load_memory_variables({}))
print("\nResumen de la conversación: \n")
print(memory.buffer)