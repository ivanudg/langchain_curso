## 0. Importar librerías
from langchain_community.document_loaders import WikipediaLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from pathlib import Path

# 1. Carga de docomentos
loader = WikipediaLoader(
    query='Lenguaje Python',
    lang='es'
)
documents = loader.load()

#print(f'Documentos cargados: {documents}')
#print(f'Número de documentos cargados: {len(documents)}')

## 2. Split de Documentos
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)
print(len(docs))

## 3. Conectar a OpenAI para los embeddings
function_embedding = OllamaEmbeddings(model='nomic-embed-text')

## 4. Incrustar documentos en BD Vectores
#ruta donde se guardará la BBDD vectorizada
persist_path = Path( Path.cwd(), 'Bloque3_Conectores_de_Datos', 'BD', 'ejemplo_wiki_bd' ) 
#print(f'Persistencia en: {persist_path}')

#Creamos la BBDD de vectores a partir de los documentos y la función embeddings
vector_store = SKLearnVectorStore.from_documents(
    documents=docs,
    embedding=function_embedding,
    persist_path=persist_path,
    serializer='parquet' #el serializador o formato de la BD lo definimos como parquet
)

# Fuerza a guardar los nuevos embeddings en el disco
vector_store.persist()

## 5a. Consulta normal similitud coseno
#Creamos un nuevo documento que será nuestra "consulta" para buscar el de mayor similitud en nuestra Base de Datos de Vectores y devolverlo
consulta = '¿Por qué el lenguaje Python se llama así?'
docs = vector_store.similarity_search(consulta)
print('Respuesta directa de la base de datos\n')
print(docs[0].page_content)
print('*'*250)

## 5b. Consulta con compresión contextual usando LLMs
from langchain_ollama import ChatOllama
from langchain_classic.retrievers import ContextualCompressionRetriever #instalar con: pip install -U langchain-classic
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

#el parámetro temperatura define la aleatoriedad de las respuestas, temperatura = 0 significa el mínimo de aleatoriedad
llm = ChatOllama(model='gpt-oss:20b')
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)
consulta_nueva = '¿Por qué el lenguaje Python se llama así?'
compressed_docs = compression_retriever.invoke(consulta_nueva)
print('Respuesta compactada por LLM de la base de datos\n')
print(compressed_docs[0].page_content)
