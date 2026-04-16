## Importar librerías iniciales e instancia de modelo de chat
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model='gpt-oss:20b-cloud')

### Conectar a BD Vectores
from langchain_community.vectorstores import SKLearnVectorStore
from pathlib import Path

Path_BBDD = Path( Path.cwd(), 'Bloque3_Conectores_de_Datos', 'BD', 'ejemplosk_embedding_db')
print(f'BBDD: {Path_BBDD}')


embbeding_function = OllamaEmbeddings(model='nomic-embed-text')
vector_store_connection = SKLearnVectorStore(
    embedding=embbeding_function,
    persist_path=Path_BBDD,
    serializer="parquet"
)

## Cargar cadena QA
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.chains.qa_with_sources import load_qa_with_sources_chain #Opción que proporciona también la fuente de datos de la respuesta

chain = load_qa_chain(
    llm,
    chain_type='stuff' #chain_type='stuff' se usa cuando se desea una manera simple y directa de cargar y procesar el contenido completo sin dividirlo en fragmentos más pequeños. Es ideal para situaciones donde el volumen de datos no es demasiado grande y se puede manejar de manera eficiente por el modelo de lenguaje en una sola operación.
)

question = "¿Qué pasó en el siglo de Oro?"
docs = vector_store_connection.similarity_search(question)
response = chain.run(
    input_documents=docs,
    question=question
)
print("Respuesta:")
print(response)
print("*"*250)

### Alternativa con método invoke
#Estructurar un diccionario con los parámetros de entrada
inputs = {
    "input_documents": docs,
    "question": question
}
response2 = chain.invoke(inputs)
print("Respuesta 2:")
print(response2["output_text"])