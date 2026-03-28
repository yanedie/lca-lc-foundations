import os
from dataclasses import dataclass
import re
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.tools import ToolRuntime, tool
from rich import print as rprint

load_dotenv()


@dataclass
class EmailContext:
    email_address: str = "julie@example.com"
    password: str = "password123"


class AuthenticatedState(AgentState):
    authenticated: bool


@tool
def check_inbox() -> str:
    """Check the inbox for recent emails"""
    return """
    Hi Julie, 
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an response email"""
    return f"Email sent to {to} with subject {subject} and body {body}"


@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the user with the given email and password"""
    if email == runtime.context.email_address and password == runtime.context.password:
        return Command(
            update={
                "authenticated": True,
                "messages": [
                    ToolMessage(
                        content="Successfully authenticated",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    else:
        return Command(
            update={
                "authenticated": False,
                "messages": [
                    ToolMessage(
                        content="Authentication failed",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )


@wrap_model_call
def dynamic_tool_call(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Allow read inbox and send email tools only if user provides correct email and password"""

    authenticated = request.state.get("authenticated")

    if authenticated:
        tools = [check_inbox, send_email]
    else:
        tools = [authenticate]

    request = request.override(tools=tools)
    return handler(request)


authenticated_prompt = (
    "You are a helpful assistant that can check the inbox and send emails."
)
unauthenticated_prompt = "You are a helpful assistant that can authenticate users."


@dynamic_prompt
def dynamic_prompt_func(request: ModelRequest) -> str:
    """Generate system prompt based on authentication status"""
    authenticated = request.state.get("authenticated")
    if authenticated:
        return authenticated_prompt
    else:
        return unauthenticated_prompt


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.3,
)

agent = create_agent(
    model=model,
    state_schema=AuthenticatedState,
    context_schema=EmailContext,
    tools=[authenticate, check_inbox, send_email],
    checkpointer=InMemorySaver(),
    middleware=[
        dynamic_tool_call,
        dynamic_prompt_func,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "authenticate": False,
                "check_inbox": False,
                "send_email": True,
            }
        ),
    ],
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="My email address is julie@example.com and the pwd is password123. Send a mail with the content draft 1 to zinyi073@gmail.com"
            )
        ]
    },
    context=EmailContext(),
    config=config,
)

rprint(response)

rprint(f"\n{"-" * 20}\n")

rprint(response["messages"][-1].content)

response = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # or "reject"
    ),
    config=config,  # Same thread ID to resume the paused conversation
)

rprint(f"\n{"-" * 20}\n")

rprint(response)

rprint(f"\n{"-" * 20}\n")

rprint(response["messages"][-1].content)
