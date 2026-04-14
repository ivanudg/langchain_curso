## Importar librerías e instancia de modelo de chat
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

chat = ChatOllama(model="gpt-oss:20b")

##  Creación objeto LLMChain
human_message_prompt = HumanMessagePromptTemplate.from_template(
    "Dame un nombre de compañía que sea simpático para una compañía que fabrique {producto}"
)

chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

from langchain_classic.chains import LLMChain #instalar con: pip install -U langchain-classic
chain = LLMChain(llm=chat, prompt=chat_prompt_template)

print(chain)

print(chain.invoke(input="Lavadoras"))

#  Cadena Secuencia Simple
from langchain_classic.chains.sequential import SimpleSequentialChain
llm = ChatOllama(model="gpt-oss:20b")
template = "Dame un simple resumen con un listado de puntos para un post de un blog acerca de {tema}"
prompt1 = ChatPromptTemplate.from_template(template)
chain_1 = LLMChain(llm=llm, prompt=prompt1)

template = "Escribe un post completo usando este resumen: {resumen}"
prompt2 = ChatPromptTemplate.from_template(template)
chain_2 = LLMChain(llm=llm, prompt=prompt2)

full_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True) #verbose=True nos irá dando paso a paso lo que hace, pudiendo ver los resultados intermedios

result = full_chain.invoke(input="Inteligencia Artificial")
print(result)