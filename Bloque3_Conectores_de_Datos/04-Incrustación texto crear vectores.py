import sys
import os

# Sube un nivel desde la carpeta actual hasta la raíz "Proyectos/"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Generar instancia de modelo de chat
from Utilities.llm_config import get_chat, get_embeddings
chat = get_chat()

## Importar librerías e instancia de modelo de chat
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

#  Incrustación de texto (embedding)
from langchain_openai import OpenAIEmbeddings

embeddings = get_embeddings()
texto = "Esto es un texto enviado a OpenAI para ser incrustado en un vector n-dimensional"
embedded_text = embeddings.embed_query(texto)
print(type(embedded_text))
print(embedded_text)

## Incrustación de documentos
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('Bloque3_Conectores_de_Datos/Fuentes datos/datos_ventas_small.csv', csv_args={'delimiter': ';'})
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



