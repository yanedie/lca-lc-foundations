import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich import print as rprint

load_dotenv()


@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"


class CustomState(AgentState):
    favourite_colour: str


@tool
def get_favourite_colour(runtime: ToolRuntime) -> str:
    "Get the favourite colour from the user"
    return runtime.context.favourite_colour


@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(
        update={
            "favourite_colour": favourite_colour,
            "messages": [
                ToolMessage(
                    content=f"Favourite colour updated to {favourite_colour} successfully!",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.3,
)

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    tools=[get_favourite_colour, update_favourite_colour],
    context_schema=ColourContext,
    state_schema=CustomState,
)

config = {"configurable": {"thread_id": "1"}}

question = HumanMessage(content="What is my favourite colour?")

response = agent.invoke({"messages": [question]}, config, context=ColourContext())

question = HumanMessage(content="My favourite colour is red.")

response = agent.invoke(
    {
        "messages": [question],
    },
    config,
)

rprint(response)
