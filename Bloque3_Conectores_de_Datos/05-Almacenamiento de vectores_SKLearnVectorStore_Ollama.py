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
#print(len(docs))

### Conectar a Omalla para los embeddings
funcion_embedding = OllamaEmbeddings(
    model='nomic-embed-text'
)

# Alternativa con SKLearn Vector Store
from langchain_community.vectorstores import SKLearnVectorStore  #pip install scikit-learn / pip install pandas pyarrow

persist_path = str(Path(pathBase / 'BD/ejemplosk_embedding_db'))

#Creamos la BBDD de vectores a partir de los documentos y la función embeddings
vector_store = SKLearnVectorStore.from_documents(
    documents=docs,
    embedding=funcion_embedding,
    persist_path=persist_path,
    serializer="parquet",  #el serializador o formato de la BD lo definimos como parquet
)

# Fuerza a guardar los nuevos embeddings en el disco
vector_store.persist()

#Creamos un nuevo documento que será nuestra "consulta" para buscar el de mayor similitud en nuestra Base de Datos de Vectores y devolverlo
consulta = 'dame información de la Primera Guerra Mundial'
docs = vector_store.similarity_search(consulta)
print('*'*250)
print(docs[0].page_content)
print('*'*250)

## Cargar la BD de vectores (uso posterior una vez tenemos creada ya la BD)
vector_store_connection = SKLearnVectorStore(
    embedding=funcion_embedding,
    persist_path=persist_path,
    serializer="parquet"
)
print('*'*250)
print(f'Una instacia de la BBDD de vectores se ha cargado desde {persist_path}')
print('*'*250)
print(vector_store_connection)
print('*'*250)

nueva_consulta = '¿Qué paso en el siglo de Oro?'
docs = vector_store_connection.similarity_search(nueva_consulta)

print('*'*250)
print(docs[0].page_content)
print('*'*250)

