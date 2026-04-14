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