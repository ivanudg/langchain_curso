## Importar librerías iniciales e instancia de modelo de chat
from langchain_core.prompts import PromptTemplate ,SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.chains import SimpleSequentialChain, LLMChain, TransformChain

#llm = ChatOllama(model='gpt-oss:20b')
llm = ChatOllama(model='gpt-oss:20b-cloud')

## Importamos documentos
from langchain_community.document_loaders import WikipediaLoader

consulta_wikipedia = input()
print(f'CONSULTA WIKIPEDIA \n {consulta_wikipedia}'+'*'*250)

idioma_final = input()
print(f'IDIOMA FINAL \n {idioma_final}'+'*'*250)

loader = WikipediaLoader(
    query=consulta_wikipedia,
    lang='es',
    load_max_docs=10
)

data = loader.load()
print(f'DATA: \n {data[0].page_content}'+'*'*250)

texto_entrada = data[0].page_content
print(f'TEXTO ENTRADA: \n {texto_entrada}'+'*'*250)

# TransformChain
### Definir la función de transformación personalizada
def transformer_function( inputs: dict ) -> dict: #Toma de entrada un diccionario y lo devuelve con la transformación oportuna
    texto = inputs['texto']
    primer_parrafo = texto.split('\n')[0]
    return {'salida': primer_parrafo}

transform_chain = TransformChain(
    input_variables = ['texto'],
    output_variables = ['salida'],
    transform = transformer_function
)

## Definir la cadena secuencial
#Creamos bloque LLMChain para resumir
template1 = "Crea un resumen en una línea del siguiente texto:\n{texto}"
prompt = ChatPromptTemplate.from_template(template1)
summary_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_key='texto_resumen'
)

#Creamos bloque LLMChain para traducir
template2 = "Traduce a"+ idioma_final + "el siguiente texto:\n{texto}"
prompt = ChatPromptTemplate.from_template(template2)
#prompt.format_prompt(idioma=idioma_final)
translate_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_key='texto_traducido'
)

sequential_chain = SimpleSequentialChain(
    chains = [transform_chain, summary_chain, translate_chain],
    verbose = True
)

result = sequential_chain(texto_entrada)

print(result)