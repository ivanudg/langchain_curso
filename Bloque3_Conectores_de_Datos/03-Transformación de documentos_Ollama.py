import sys
from pathlib import Path

## Transformador "Character Text Splitter"
from langchain_text_splitters import CharacterTextSplitter

#  Carga del fichero
rutaBase = Path.cwd()
insumos = Path(rutaBase / 'Fuentes datos')

#with open('Bloque3_Conectores_de_Datos/Fuentes datos/Historia España.txt', encoding='utf8') as file:
archivo = Path(insumos / 'Historia Espania.txt')
with open(archivo, encoding='utf8') as file:
    texto_completo = file.read()

# Números de caracteres
print(len(texto_completo))
# Número de palabras
print(len(texto_completo.split()))

## Transformador "Character Text Splitter"
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000) #Indicamos que divida cuando se encuentra 1 salto de línea y trate de hacer fragmentos de 1000 caracteres
texts = text_splitter.create_documents([texto_completo]) #Creamos documentos gracias al transformador

print(len(texts)) #Número de fragmentos creados
print(type(texts))#Verificamos el tipo del objeto obtenido
print('\n')
print(type(texts[0])) #Verificamos el tipo de cada elemento
print('\n')
print(texts[0])
print(len(texts[0].page_content)) #Número de caracteres del primer fragmento
print(texts[1])
print(len(texts[1].page_content)) #Número de caracteres del segundo fragmento