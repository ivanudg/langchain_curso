## Importar librerías iniciales e instancia de modelo de chatfrom langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_classic.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor

# Instanciar un modelo
llm = ChatOllama(
    model='gemma4:31b-cloud',
    temperature=0
)

# CASO DE USO 1
from langchain_classic.memory import ConversationBufferMemory

#ponemos una denominada clave a la memoria "chat_history"
memory = ConversationBufferMemory(memory_key='chat_history')
tools = load_tools(['wikipedia', 'llm-math'], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors= True
)
agent.invoke("Dime 5 productos esenciales para el mantenimiento del vehículo.")
agent.invoke("¿Cuál de los anteriores es el más importante?")
agent.invoke("Necesito la respuesta anterior en castellano")