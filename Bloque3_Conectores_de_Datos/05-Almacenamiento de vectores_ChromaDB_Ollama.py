## Importar librerías
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

### Carga de documento y split
from pathlib import Path

pathBase = Path.cwd()
archivo = str(Path(pathBase / 'Fuentes datos/Historia Espania.txt'))

# Cargar el documento
loader = TextLoader(archivo, encoding='utf8')
documents = loader.load()

# Dividir en chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)  #Otro método de split basándose en tokens
docs = text_splitter.split_documents(documents)

### Conectar a Omalla para los embeddings
funcion_embedding = OllamaEmbeddings(
    model='nomic-embed-text'
)

# Alternativa con ChromaDB
import chromadb #pip install chromadb en una terminal
from langchain_chroma import Chroma #pip install langchain_chroma en una terminal

# Cargar en ChromaDB
persist_path = str(Path(pathBase / 'BD/ejemploChromaDB_embedding_db'))

db = Chroma.from_documents(
    docs,
    funcion_embedding,
    collection_name="langchain",
    persist_directory=persist_path
)
#Se crean en el directorio persistente la carpeta con los vectores y otra con las string, aparte de una carpeta "index" que mapea vectores y strings

### Cargar los Embeddings desde el disco creando la conexión a ChromaDB
db_connection = Chroma(
    persist_directory=persist_path,
    embedding_function=funcion_embedding
)

#Creamos un nuevo documento para buscar el de mayor similitud en nuestra Base de Datos de Vectores y devolverlo
nuevo_documento = "What did FDR say about the cost of food law?"
docs = db_connection.similarity_search(nuevo_documento)
print('*'*250)
print(docs[0].page_content) #El primer elemento es el de mayor similitud, por defecto se devuelven hasta 4 vectores (k=4)
print('*'*250)