import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from rich import print as rprint

load_dotenv()


def image_to_base64(image_path_str: str) -> str:
    path = Path(image_path_str)

    if not path.is_file():
        raise FileNotFoundError(f"找不到图片文件: {path}")

    binary_data = path.read_bytes()
    base64_str = base64.b64encode(binary_data).decode("utf-8")

    return base64_str


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="deepseek-chat",
    temperature=0.3,
)

agent = create_agent(
    model=model,
)

img_b64 = image_to_base64("capital.png")

multimodal_question = HumanMessage(
    content=[
        {"type": "text", "text": "Tell me about this capital"},
        {"type": "image", "base64": img_b64, "mime_type": "image/png"},
    ]
)


response = agent.invoke(
    {"messages": [multimodal_question]},
)

rprint(response["messages"][-1].content)
