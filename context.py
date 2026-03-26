import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from rich import print as rprint

load_dotenv()


@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"


@tool
def get_favourite_colour(runtime: ToolRuntime) -> str:
    "Get the favourite colour from the user"
    return runtime.context.favourite_colour


@tool
def get_least_favourite_colour(runtime: ToolRuntime) -> str:
    "Get the least favourite colour from the user"
    return runtime.context.least_favourite_colour


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.9,
)

agent = create_agent(
    model=model,
    tools=[get_favourite_colour, get_least_favourite_colour],
    context_schema=ColourContext,
)

question = HumanMessage(content="What is your favourite colour?")

# AI不是user，所有访问不了上下文
response = agent.invoke({"messages": [question]}, context=ColourContext())

question = HumanMessage(content="What is my favourite colour?")

# AI访问上下文，获取用户的喜好，如果没有提供上下文，则使用默认值
response = agent.invoke(
    {"messages": [question]}, context=ColourContext(favourite_colour="red")
)

rprint(response)
