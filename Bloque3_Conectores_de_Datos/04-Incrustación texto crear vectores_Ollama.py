from pathlib import Path

# Generar instancia de modelo de chat
from langchain_ollama import ChatOllama
chat = ChatOllama(
    model='gpt-oss:20b'
)

## Importar librerías e instancia de modelo de chat
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

#  Incrustación de texto (embedding)
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(
    model='nomic-embed-text'
)

texto = "Esto es un texto enviado a OpenAI para ser incrustado en un vector n-dimensional"
embedded_text = embeddings.embed_query(texto)
print(type(embedded_text))
print(embedded_text)

## Incrustación de documentos
from langchain_community.document_loaders import CSVLoader

pathBase = Path.cwd()
archivo = str(Path(pathBase / 'Fuentes datos/datos_ventas_small.csv'))
#loader = CSVLoader('Bloque3_Conectores_de_Datos/Fuentes datos/datos_ventas_small.csv', csv_args={'delimiter': ';'})
loader = CSVLoader(archivo, csv_args={'delimiter': ';'})
data = loader.load()
print(type(data))
print(data[0])

#No podemos incrustar el objeto "data" puesto que es una lista de documentos, lo que espera es una string
#embedded_docs = embeddings.embed_documents(data)

#Creamos una comprensión de listas concatenando el campo "page_content" de todos los documentos existentes en la lista "data"
print([elemento.page_content for elemento in data])

embedded_docs = embeddings.embed_documents([elemento.page_content for elemento in data])
#Verificamos cuántos vectores a creado (1 por cada registro del fichero CSV con datos)
print(len(embedded_docs))

#Vemos un ejemplo del vector creado para el primer registro
print(embedded_docs[1])



