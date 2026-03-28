import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from openai import base_url

load_dotenv()


@dataclass
class LanguageContext:
    user_language: str = "English"


@dynamic_prompt
def user_language_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_language = request.runtime.context.user_language
    base_prompt = "You are a helpful assistant."
    if user_language != "English":
        return f"{base_prompt} only respond in {user_language}."
    elif user_language == "English":
        return base_prompt


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.3,
)

agent = create_agent(
    model=model, context_schema=LanguageContext, middleware=[user_language_prompt]
)

response = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="Chinse"),
)

print(response["messages"][-1].content)

response = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="English"),
)

print(response["messages"][-1].content)

response = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="Japanese"),
)

print(response["messages"][-1].content)
