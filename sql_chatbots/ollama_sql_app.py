import os
LANGCHAIN_API_KEY = os.getenv(key="LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv(key="LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv(key="LANGCHAIN_TRACING_V2")
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.pydantic_v1 import BaseModel

# llms
#  we are using Gemma:2b llm
phi_llm = ChatOllama(model='phi',temperature=0.1,timeout=300)
gemma_llm = ChatOllama(model='gemma:2b',temperature=0.1,timeout=300)

# connect to sqlite db
db = SQLDatabase.from_uri("sqlite:///nba_roster.db", sample_rows_in_table_info= 0)
# query the db
# query = "SELECT Team FROM nba_roster WHERE NAME = 'Klay Thompson'"
# res = db.run(query)
# print(res)

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

# prompt
template = """Based on the table schema below, return only a SQL query that would answer the user's question. No pre-amble.:
{schema}

Question: {question}
Answer:"""  # noqa: E501


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
        MessagesPlaceholder(variable_name="history"),
        ("human", template),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

# we need to format the gemma llm output
def format_query(query):
    return query.strip().replace("```sql", "").replace("```", "")

# Chain to query with memory
sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
        history=RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]),
    )
    | prompt
    # | llm.bind(stop=["\nUser:"])
    # | gemma_llm.bind(stop=["\n"])
    # | llm.bind(stop=["\nUser:", "\nRules:", "\nAssistant"])
    # | gemma_llm.bind(stop=["\nSQLResult:"])
    # | phi_llm
    | gemma_llm
    | StrOutputParser()
    | RunnableLambda(format_query)
)


# print(sql_chain.invoke({"question": "What team is Klay Thompson on?"}))
# print(sql_chain.invoke({"question": "Give me total number of players in Golden State Warriors Team?"}))
# print(sql_chain.invoke({"question": "How many total different teams are there in database?"}))
# print(sql_chain.invoke({"question": "give me name of all players who's age is 19 years?"}))

def save(input_output):
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]

sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save

# Chain to answer
template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""  # noqa: E501

prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural "
            "language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)

# Supply the input types to the prompt
class InputType(BaseModel):
    question: str

# final chain 
chain = (
    RunnablePassthrough.assign(query=sql_response_memory).with_types(
        input_type=InputType
    )
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | phi_llm
)

print(chain.invoke({"question": "How many total different teams are there in nba roaster?"}))

