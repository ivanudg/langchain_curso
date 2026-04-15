## Importar librerías
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

### Carga de documento y split
from pathlib import Path

pathBase = Path.cwd()
archivo = str(Path(pathBase / 'Fuentes datos/Historia Espania.txt'))
#print(archivo)

# Cargar el documento
loader = TextLoader(archivo, encoding='utf8')
documents = loader.load()

# Dividir en chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)  #Otro método de split basándose en tokens
docs = text_splitter.split_documents(documents)
print(len(docs))

### Conectar a Omalla para los embeddings
function_embeddings = OllamaEmbeddings(
    model='nomic-embed-text'
)

# Alternativa con SKLearn Vector Store
from langchain_community.vectorstores import SKLearnVectorStore  #pip install scikit-learn / pip install pandas pyarrow

persist_path = str(Path(pathBase / 'BD'))

#Creamos la BBDD de vectores a partir de los documentos y la función embeddings
vector_store = SKLearnVectorStore.from_documents(
    documents=docs,
    embedding=function_embeddings,
    persist_path=persist_path,
    serializer="parquet",  #el serializador o formato de la BD lo definimos como parquet
)

# Fuerza a guardar los nuevos embeddings en el disco
vector_store.persist()