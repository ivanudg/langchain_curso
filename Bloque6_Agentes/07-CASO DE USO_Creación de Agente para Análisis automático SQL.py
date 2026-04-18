## 0.Importar librerías iniciales e instancia de modelo de chat

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_classic.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor

#Generar instancia
llm = ChatOllama(model='gemma4:31b-cloud', temperature=0)

