import langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import os
from dotenv import load_dotenv

# Obtener la clave
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(openai_api_key=api_key)

# Obtener 1 resultado invocando al chat de OpenAI
# resultado = chat.invoke([HumanMessage(content="¿Puedes decirme dónde se encuentra Cáceres?")])
# print("Resultado de ChatGPT")
# print(resultado)
# print("Resultado.content de ChatGPT")
# print(resultado.content)

# Obtener 2 resultado pero ahora colocando como debe comportarsé el agente con SystemMesage
# resultado = chat.invoke(
#     [
#         SystemMessage(content='Eres un historiador que conoce los detalles de todas las ciudades del mundo'),
#         HumanMessage(content='¿Puedes decirme dónde se encuentra Cáceres')
#     ]
# )

# print("Resultado.content de ChatGPT")
# print(resultado.content)

# Obtener varios resultados invocando al chat de OpenAI con "generate"
resultado = chat.generate(
    [
        [
            SystemMessage(content='Eres un historiador que conoce los detalles de todas las ciudades del mundo'),
            HumanMessage(content='¿Puedes decirme dónde se encuentra Cáceres')
        ],
        [
                SystemMessage(content='Eres un joven rudo que no le gusta que le pregunten, solo quiere estar de fiesta'),
                HumanMessage(content='¿Puedes decirme dónde se encuentra Cáceres')
        ]
    ]
)

print("Obtener primer resultado")
print(resultado.generations[0][0].text)
print("Obtener segundo resultado")
print(resultado.generations[1][0].text)