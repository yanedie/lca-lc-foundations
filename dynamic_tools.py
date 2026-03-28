import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain.messages import HumanMessage
from langchain.tools import tool
from openai import base_url
from tavily import TavilyClient

# web search
tavily_client = TavilyClient()

# database
db = SQLDatabase.from_uri("sqlite:///notebooks/module-3/resources/Chinook.db")


# runtime context
@dataclass
class UserRole:
    user_role: str = "external"


# tools
@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)


@tool
def sql_query(query: str) -> str:
    """Obtain information from the database using SQL queries"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


@wrap_model_call
def dynamic_tool_call(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Dynamically call tools based on the runtime context"""

    user_role = request.runtime.context.user_role

    if user_role == "internal":
        pass  # internal users get access to all tools
    else:
        tools = [web_search]  # external users only get access to web search
        request = request.override(tools=tools)

    return handler(request)


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.3,
)

agent = create_agent(
    model=model,
    tools=[web_search, sql_query],
    middleware=[dynamic_tool_call],
    context_schema=UserRole,
)

response = agent.invoke(
    {"messages": [HumanMessage(content="How many artists are in the database?")]},
    context={"user_role": "external"},
)

print(response["messages"][-1].content)
