import os

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from rich import print as rprint
from langgraph.types import Command

load_dotenv()


@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the given address."""
    # take email from state
    return runtime.state["email"]


@tool
def send_email(body: str) -> str:
    """Send an email to the given address with the given subject and body."""
    # fake email sending
    return f"Email sent"


class EmailState(AgentState):
    email: str


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.3,
)

agent = create_agent(
    model=model,
    tools=[read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_email": False,
                "send_email": True,
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Please read my email and send a response immediately. Send the reply now in the same thread."
            )
        ],
        "email": "Hi Seán, I'm going to be late for our meeting tomorrow. Can we reschedule? Best, John.",
    },
    config=config,
)

# 批准工具执行
# response = agent.invoke(
#     Command(resume={"decisions": [{"type": "approve"}]}),
#     config=config,  # Same thread ID to resume the paused conversation
# )

# 拒绝工具执行
response = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    # An explanation of why the request was rejected
                    "message": "No please sign off - Your merciful leader, Seán.",
                }
            ]
        }
    ),
    config=config,  # Same thread ID to resume the paused conversation
)

print("-" * 20)

# 编辑工具执行
response = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    # Edited action with tool name and args
                    "edited_action": {
                        # Tool name to call.
                        # Will usually be the same as the original action.
                        "name": "send_email",
                        # Arguments to pass to the tool.
                        "args": {"body": "This is the last straw, you're fired!"},
                    },
                }
            ]
        }
    ),
    config=config,  # Same thread ID to resume the paused conversation
)

rprint(response)

print("-" * 20)

if hasattr(response, "__interrupt__"):
    print(response["__interrupt__"][0].value["action_requests"][0]["args"]["body"])

rprint(response["messages"][-1].content)
