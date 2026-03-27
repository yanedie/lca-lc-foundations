import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich import print as rprint

load_dotenv()

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
    middleware=[
        SummarizationMiddleware(
            model=model, trigger=("tokens", 100), keep=("messages", 1)
        )
    ],
)


response = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is the capital of the moon?"),
            AIMessage(content="The capital of the moon is Lunapolis."),
            HumanMessage(content="What is the weather in Lunapolis?"),
            AIMessage(
                content="Skies are clear, with a high of 120C and a low of -100C."
            ),
            HumanMessage(content="How many cheese miners live in Lunapolis?"),
            AIMessage(content="There are 100,000 cheese miners living in Lunapolis."),
            HumanMessage(content="Do you think the cheese miners' union will strike?"),
            AIMessage(content="Yes, because they are unhappy with the new president."),
            HumanMessage(
                content="If you were Lunapolis' new president how would you respond to the cheese miners' union?"
            ),
        ]
    },
    {"configurable": {"thread_id": "1"}},
)

rprint(response)
