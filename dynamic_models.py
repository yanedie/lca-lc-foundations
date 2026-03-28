from email import message
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

load_dotenv()

large_model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="Qwen3.5-397B-A17B-FP8",
    temperature=0.3,
)

standard_model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="Qwen3-VL-32B-Instruct",
    temperature=0.3,
)


@wrap_model_call
def state_based_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelRequest:
    """Select model based on State conversation length."""
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)

    if message_count > 10:
        model = large_model
    else:
        model = standard_model

    request = request.override(model=model)

    return handler(request)


agent = create_agent(
    model=large_model,
    middleware=[state_based_model],
    system_prompt="You are roleplaying a real life helpful office intern.",
)

question = HumanMessage(content="Did you water the office plant today?")

response = agent.invoke({"messages": [question]})

print(response["messages"][-1].content)

print(response["messages"][-1].response_metadata["model_name"])

response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Did you water the office plant today?"),
            AIMessage(content="Yes, I gave it a light watering this morning."),
            HumanMessage(content="Has it grown much this week?"),
            AIMessage(content="It's sprouted two new leaves since Monday."),
            HumanMessage(content="Are the leaves still turning yellow on the edges?"),
            AIMessage(content="A little, but it's looking healthier overall."),
            HumanMessage(
                content="Did you remember to rotate the pot toward the window?"
            ),
            AIMessage(
                content="I rotated it a quarter turn so it gets more even light."
            ),
            HumanMessage(content="How often should we be fertilizing this plant?"),
            AIMessage(
                content="About once every two weeks with a diluted liquid fertilizer."
            ),
            HumanMessage(content="When should we expect to have to replace the pot?"),
        ]
    }
)

print(response["messages"][-1].content)

print(response["messages"][-1].response_metadata["model_name"])
