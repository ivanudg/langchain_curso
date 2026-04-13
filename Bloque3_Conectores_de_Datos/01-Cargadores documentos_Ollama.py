## Importar librerías e instancia de modelo de chat
import sys

from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
## Carga datos
from langchain_community.document_loaders import (
    CSVLoader, # Carga datos CSV
    BSHTMLLoader, # Carga datos HTML
    PyPDFLoader # Carga datos PDF
)
from pathlib import Path

# Generar instancia de modelo de chat
chat = ChatOllama(
    model='gpt-oss:20b'
)

# Ruta Base
rutaBase = Path.cwd()
insumos = Path(rutaBase / "Fuentes datos")
print(f'Ruta Base: {rutaBase}')
print(f'Ruta Base: {insumos}')

#Cargamos el fichero CSV
loader = CSVLoader(
    file_path=Path(insumos / "datos_ventas_small.csv"),
    csv_args={"delimiter": ";"}
)

#Creamos el objeto "data" con los datos desde el cargador "loader"
data = loader.load()
print("*"*150+'\n')
print("CARGA DE DATOS CSV")
print("*"*150+'\n')
print(type(data)) #Mostramos el tipo de dato del objeto "data"
print("*"*150+'\n')
print(data[0])
print("*"*150+'\n')
print(data[1])
print("*"*150+'\n')
print(data[1].page_content)


## Carga datos HTML
#loader_html = BSHTMLLoader(file_path="Bloque3_Conectores_de_Datos/Fuentes datos/ejemplo_web.html")
loader_html = BSHTMLLoader(file_path=Path( insumos / "ejemplo_web.html"))
data_html = loader_html.load()
print("*"*150)
print("CARGA DE DATOS HTML")
print("*"*150+'\n')
print(data_html) #Mostramos el tipo de dato del objeto "data_html"
print("*"*150)
print(data_html[0].page_content)

## Carga datos PDF
#loader_pdf = PyPDFLoader('Bloque3_Conectores_de_Datos/Fuentes datos/Documento tecnologías emergentes.pdf')
loader_pdf = PyPDFLoader(str(Path( insumos / 'Documento tecnologías emergentes.pdf')))
pages_pdf = loader_pdf.load_and_split() # Carga y divide el PDF en páginas
print("*"*150)
print("CARGA DE DATOS PDF")
print("*"*150+'\n')
print(type(pages_pdf)) #Mostramos el tipo de dato del objeto "pages_pdf"
print("*"*150)
print(pages_pdf[0]) #Mostramos el contenido de la primera página del PDF
print("*"*150+'\n')
print(pages_pdf[0].page_content) #Mostramos el contenido de la primera página del PDF

# Caso de uso: Resumir PDFs
contenido_pdf = pages_pdf[0].page_content # Extraemos el contenido de la primera página del PDF

print("*"*150)
print("CASO DE USO: RESUMIR PDF")
print("*"*150+'\n')
print("*"*150)
print("Contenido de la primera página del PDF:")
print("*"*150+'\n')
print(contenido_pdf)

human_template = '"Necesito que hagas un resumen del siguiente texto: \n{contenido}"'
human_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
format_message = chat_prompt.format_prompt(contenido=contenido_pdf)

print("*"*150)
print("Format Message:")
print("*"*150+'\n')
print(format_message)

solicitud_completa = format_message.to_messages()
respuesta = chat.invoke(solicitud_completa)
print("*"*150)
print("Resumen primera página:")
print("*"*150+'\n')
print(respuesta.content)

#Resumir el documento completo
#Creamos una string concatenando el contenido de todas las páginas
documento_completo = "\n".join([pagina.page_content for pagina in pages_pdf])
print("*"*150)
print("Documento completo:")
print("*"*150+'\n')
print(documento_completo)

solicitud_completa = chat_prompt.format_prompt(contenido=documento_completo).to_messages()
respuesta = chat.invoke(solicitud_completa)
print("*"*150)
print("Resumen documento completo:")
print("*"*150+'\n')
print(respuesta.content)