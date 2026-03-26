import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from rich import print as rprint

load_dotenv()


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x**0.5


@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x**2


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.3,
)

subagent_1 = create_agent(model=model, tools=[square_root])

subagent_2 = create_agent(model=model, tools=[square])


@tool
def call_subagent_1(x: float) -> float:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke(
        {"messages": [HumanMessage(content=f"Calculate the square root of {x}")]}
    )
    return response["messages"][-1].content


@tool
def call_subagent_2(x: float) -> float:
    """Call subagent 2 in order to calculate the square of a number"""
    response = subagent_2.invoke(
        {"messages": [HumanMessage(content=f"Calculate the square of {x}")]}
    )
    return response["messages"][-1].content


main_agent = create_agent(
    model=model,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a helpful assistant who can call subagents to calculate the square root or square of a number.",
)

question = "What is the square root of 456?"

response = main_agent.invoke({"messages": [HumanMessage(content=question)]})

rprint(response)
