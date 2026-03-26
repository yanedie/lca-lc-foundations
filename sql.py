import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.messages import HumanMessage
from rich import print as rprint


load_dotenv()
db = SQLDatabase.from_uri("sqlite:///notebooks/module-2/resources/Chinook.db")


@tool
def sql_query(query: str) -> str:
    """Obtain information from the database using SQL queries"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.3,
)

agent = create_agent(model=model, tools=[sql_query])


question = HumanMessage(
    content="Who is the most popular artist beginning with 'S' in this database?"
)

response = agent.invoke({"messages": [question]})

rprint(response["messages"])
