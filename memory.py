import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich import print as rprint

load_dotenv()


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="deepseek-chat",
    temperature=0.3,
)

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
)

question = HumanMessage(
    content="Hello my name is Seán and my favourite colour is green"
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [question]},
    config,
)

rprint(response["messages"])

question = HumanMessage(content="What's my favourite colour?")

response = agent.invoke(
    {"messages": [question]},
    config,
)

rprint(response)
