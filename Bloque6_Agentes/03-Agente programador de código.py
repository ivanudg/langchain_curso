## Importar librerías iniciales e instancia de modelo de chat
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_classic.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor

## Instanciar modelo de chat
#llm = ChatOllama(model="gpt-oss:20b-cloud", temperature=0) #Recomendable temperatura a 0 para que el LLM no sea muy creativo, vamos a tener muchas herramientas a nuestra disposición y queremos que sea más determinista
llm = ChatOllama(model="gemma4:31b-cloud", temperature=0) #Recomendable temperatura a 0 para que el LLM no sea muy creativo, vamos a tener muchas herramientas a nuestra disposición y queremos que sea más determinista

from langchain_experimental.agents.agent_toolkits import create_python_agent #pip install -U langchain-experimental
from langchain_experimental.tools.python.tool import PythonREPLTool

import os
os.system('clear' if os.name != 'nt' else 'cls')

## Creamos el agente para crear y ejecutar código Python 
agent = create_python_agent(
    tool=PythonREPLTool(),
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

lista_ejemplo = [3,1,5,3,5,6,7,3,5,10]

#agent.invoke(f'''ordena la lista {lista_ejemplo}''')

## Ejemplo con un dataframe
import pandas as pd
from pathlib import Path

csv = Path( Path.cwd(), 'Bloque6_Agentes', 'insumos', 'datos_ventas_small.xlsx')
df = pd.read_excel( str(csv) )
print('*'*200)
print(df.head())
print('*'*200+'\n')

#agent.invoke(f'''¿Qué sentencias de código tendría que ejecutar para obtener la suma de venta total agregada por Línea de Producto? Este sería el dataframe {df}, no tienes que ejecutar la sentencia, solo pasarme el código a ejecutar''')

print('*'*200)
print(df.groupby('Línea Producto')['Venta total'].sum())
print('*'*200+'\n')

#agent.invoke(f'''¿Cuál es la suma agregada de la venta total para la línea de proudcto "Motorcycles"? Este sería el dataframe {df}''')

#agent.invoke(f'''¿Qué sentencias de código tendría que ejecutar para tener una visualización con la librería Seaborn que agregue a nivel de Línea de Producto el total de venta? Este sería el dataframe {df}, recuerda que no tienes que ejecutar la sentencia, solo pasarme el código a ejecutar''')

#import seaborn as sns
#sns.barplot(x='Línea Producto', y='Venta total', data=df, estimator=sum)