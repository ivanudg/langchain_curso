## 0.Importar librerías iniciales e instancia de modelo de chat
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

## 1.Cargamos la BD Vectores y el compresor
#Podríamos establecer que tuviera memoria
from langchain_classic.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history") #ponemos una denominada clave a la memoria "chat_history"

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from pathlib import Path

funcion_embedding = OllamaEmbeddings(
    model='nomic-embed-text'
)
persist_path = Path( Path.cwd(), 'Bloque3_Conectores_de_Datos', 'BD', 'ejemplosk_embedding_db' )
vector_store_connection = SKLearnVectorStore(
    embedding=funcion_embedding,
    persist_path=persist_path,
    serializer='parquet'
)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store_connection.as_retriever()
)

## 2.Creamos una nueva herramienta a partir de la BD Vectorial para obtener resultados optimizados
from langchain_classic.agents import tool

@tool
def consulta_interna(text: str) -> str:
    '''Retorna respuestas sobre la historia de España. Se espera que la entrada sea una cadena de texto
    y retorna una cadena con el resultado más relevante. Si la respuesta con esta herramienta es relevante,
    no debes usar ninguna herramienta más ni tu propio conocimiento como LLM'''
    compressed_docs = compression_retriever.invoke(text)
    resultado = compressed_docs[0].page_content
    
    # ✅ Validar antes de acceder
    if not compressed_docs:
        return "No se encontró información relevante en la base de datos interna."
    
    resultado = compressed_docs[0].page_content
    return resultado

tools = load_tools(["wikipedia","llm-math"],llm=llm)
tools=tools+[consulta_interna]

## 3.Creamos el agente y lo ejecutamos 
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors= True
)

agent.invoke("¿Qué periodo abarca cronológicamente en España el siglo de oro?")
agent.invoke("¿Qué pasó durante la misma etapa en Francia?") #Gracias a tener memoria compara en esas fechas qué ocurrió en Francia
agent.invoke("¿Cuáles son las marcas de vehículos más famosas hoy en día?") #Pregunta que no podemos responder con nuestra BD Vectorial
