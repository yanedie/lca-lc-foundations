import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich import print as rprint

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "local_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["notebooks/module-2/resources/2.1_mcp_server.py"],
            }
        }
    )

    # get tools
    tools = await client.get_tools()

    # get resources
    resources = await client.get_resources("local_server")

    # get prompts
    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content

    model = init_chat_model(
        model_provider="openai",
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-5-nano",
        temperature=0.3,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
    )

    config = {"configurable": {"thread_id": "1"}}

    question = HumanMessage(content="Tell me about the langchain-mcp-adapters library")

    response = await agent.ainvoke({"messages": [question]}, config=config)

    rprint(response)
    rprint(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
