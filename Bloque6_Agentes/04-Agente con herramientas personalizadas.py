## Importar librerías iniciales e instancia de modelo de chat
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_classic.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor

import os
os.system('clear' if os.name != 'nt' else 'cls')

# Instanciar un modelo
llm = ChatOllama(
    model='gemma4:31b-cloud',
    temperature=0
)

# CASO DE USO 1
## Creamos nuestra herramienta personalizada 
from langchain_classic.agents import tool

@tool
def persona_amable(text:str) -> str:
    '''Retorna la persona más amable. Se espera que la entrada esté vacía "" 
    y retorna la persona más amable del universo'''
    return  "Miguel Celebres"

## Definimos las herramientas a las que tendrá acceso el agente y ejecutamos
tools = load_tools(["wikipedia","llm-math",],llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors= True
)

#agent.invoke("¿Quién es la persona más amable del universo?")

tools = tools + [persona_amable]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors= True
)

#agent.invoke("¿Quién es la persona más amable del universo?")

# CASO DE USO 3: Consultar hora actual
# Solicitud con las herramientas actuales no proporciona el resultado que queremos
#agent.invoke("¿Cuál es la hora actual?")

## Creamos nuestra función personalizada 
from datetime import datetime
@tool
def hora_actual(text: str)->str:
    '''Retorna la hora actual, debes usar esta función para cualquier consulta sobre la hora actual. Para fechas que no sean
    la hora actual, debes usar otra herramienta. La entrada está vacía y la salida retorna una string'''
    return str(datetime.now())

tools = tools + [hora_actual]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors= True
)

# Solicitud con las herramientas actuales SÍ proporciona el resultado que queremos
agent.invoke("¿Cuál es la hora actual?")