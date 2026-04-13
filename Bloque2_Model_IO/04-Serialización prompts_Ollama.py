## Importar librerías e instancia de modelo de chat
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, load_prompt
from langchain_ollama import ChatOllama

### Cargar plantilla prompt -> Se comentó debido a que se importa en la parte superior
# y así no duplicar líneas
# from langchain_core.prompts import load_prompt

### Guardar plantilla prompt
plantilla = "Pregunta: {pregunta_usuario}\n\nRespuesta: Vamos a verlo paso a paso."
prompt = PromptTemplate(template=plantilla)
prompt.save("Bloque1_Model_IO/plantillas_prompts/prompt.json") #Guardamos la plantilla en un archivo JSON

prompt_cargado = load_prompt("Bloque1_Model_IO/plantillas_prompts/prompt.json") #Cargamos la plantilla desde el archivo JSON
print(prompt_cargado) #Mostramos la plantilla cargada para verificar que se ha cargado correctamente    
#print(prompt_cargado.format(pregunta_usuario="¿Cuál es la capital de Francia?")) #Probamos la plantilla cargada con un ejemplo de pregunta