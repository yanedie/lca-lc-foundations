import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from rich import print as rprint

load_dotenv()


@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x**0.5


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="deepseek-chat",
    temperature=0.3,
)

# response = model.invoke("What's the capital of the Moon?")
# print(response.content)

agent = create_agent(
    model=model,
    tools=[tool1],
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
)

question = HumanMessage(content="What is the square root of 467?")

response = agent.invoke({"messages": [question]})

rprint(response["messages"])
rprint(response["messages"][1].tool_calls)
