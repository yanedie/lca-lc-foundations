import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from pydantic import BaseModel
from rich import print as rprint

load_dotenv()


class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str


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
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
    response_format=CapitalInfo,
)

question = HumanMessage(content="What is the capital of The Moon?")

response = agent.invoke({"messages": [question]})

rprint(response["structured_response"])

print("-" * 80)
# 假设 msg 是你的最后一个消息对象
msg = response["messages"][-1]
print(f"1. 【内容】 content: \n{msg.content}\n")
print(f"2. 【类型】 type: {msg.type}\n")
print(f"3. 【ID】 id: {msg.id}\n")
print(f"4. 【原始元数据】 response_metadata: {msg.response_metadata}\n")
print(f"5. 【额外参数】 additional_kwargs: {msg.additional_kwargs}\n")

# 工具类型的消息没有以下两个属性
if hasattr(msg, "usage_metadata"):
    print(f"6. 【Token 统计】 usage_metadata: {msg.usage_metadata}")
if hasattr(msg, "tool_calls"):
    print(f"7. 【工具调用】 tool_calls: {msg.tool_calls}\n")
