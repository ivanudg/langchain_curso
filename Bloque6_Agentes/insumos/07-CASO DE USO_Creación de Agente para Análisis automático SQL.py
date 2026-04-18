## 0.Importar librerías iniciales e instancia de modelo de chat

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_classic.agents import load_tools,initialize_agent,AgentType,create_react_agent,AgentExecutor

#Generar instancia
llm = ChatOllama(model='gemma4:31b-cloud', temperature=0)

#pip install mysql-connector-python
import mysql.connector
import os
import dotenv

dotenv.load_dotenv()
pass_sql = os.getenv('MY_SQL_Pass')
os.system('clear' if os.name != 'nt' else 'cls')

# Configuración de la conexión a la base de datos
config = {
    'user': 'root',       
    'password': pass_sql, 
    'host': '127.0.0.1',         
    'database': 'world'
}

# Conectar a la base de datos
conn = mysql.connector.connect(**config)
cursor = conn.cursor()

# 2. Ejecutamos consulta manualmente (sin agentes Langchain)
# Definir la consulta manualmente: tengo una base de datos mysql en mi computadora local denominada "world" y una tabla "Country" 
#sobre la que quiero hacer la suma de la población en la columna "Population" para el continente Asia (columna "Continent")
query = """
    SELECT SUM(Population)
    FROM Country
    WHERE Continent = 'Asia';
    """

# Ejecutar la consulta
cursor.execute(query)
result = cursor.fetchone()
suma_poblacion = result[0] if result[0] is not None else 0
print(f"La suma de la población del continente Asia es: {suma_poblacion}")

## 3.Creamos el agente SQL 
#from langchain_community.agent_toolkits import create_sql_agent
from langchain_classic.agents import create_sql_agent
#from langchain_classic.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_classic.agents.agent_types import AgentType

# Crear una cadena de conexión a la base de datos MySQL
connection_string = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}/{config['database']}"

# Crear una instancia de la base de datos SQL
db = SQLDatabase.from_uri(connection_string)

def handle_error(error) -> str:
    return str(error)[:500]  # retorna el error truncado para que el agente reintente


# agent = create_sql_agent(
#     llm,
#     db=db,
#     verbose=True,
#     #handle_parsing_errors=True
#     handle_parsing_errors=handle_error
# )

agent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_executor_kwargs={
        "handle_parsing_errors": True  # ← así se pasa en create_sql_agent
    }
)

agent.invoke("Dime la población total de Asia")
result = agent.invoke("Dime el promedio de la esperanza de vida por cada una de las regiones ordenadas de mayor a menor")

# Mostrar el resultado
print("\nSalida:\n")
print(result["output"])