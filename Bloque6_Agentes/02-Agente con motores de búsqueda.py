from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_classic.agents import load_tools, initialize_agent, AgentType, create_react_agent, AgentExecutor

# Crear instancia del modelo de lenguaje
llm = ChatOllama(model="gpt-oss:20b-cloud", temperature=0)

## Definir SERP API Key
import os
import dotenv

dotenv.load_dotenv()

get_serpapi_key = os.getenv("SERPAPI_KEY")
os.environ["SERPAPI_API_KEY"] = get_serpapi_key

## Definimos las herramientas a las que tendrá acceso el agente
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    #agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
agent.invoke("¿En qué año nació Einstein? ¿Cuál es el resultado de ese año multiplicado por 3?")