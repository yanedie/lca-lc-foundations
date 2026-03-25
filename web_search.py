import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from rich import print as rprint
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)


web_search.invoke("Who is the current mayor of San Francisco?")


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="deepseek-chat",
    temperature=0.3,
)

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
)

question = HumanMessage(content="Who is the current mayor of San Francisco?")

response = agent.invoke({"messages": [question]})

rprint(response["messages"])
