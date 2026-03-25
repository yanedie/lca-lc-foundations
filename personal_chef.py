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


system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

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
    system_prompt=system_prompt,
)

question = HumanMessage(
    content="我有土豆、胡萝卜、玉米、鸡毛菜和葱。我可以做什么晚饭？"
)

response = agent.invoke({"messages": [question]})

rprint(response)
rprint(response["messages"][-1].content)
