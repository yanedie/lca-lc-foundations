import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print as rprint

load_dotenv()


loader = PyPDFLoader("notebooks/module-2/resources/IFU.pdf")

data = loader.load()

rprint(len(data))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=80, add_start_index=True
)

all_splits = text_splitter.split_documents(data)

print(len(all_splits))

embeddings = OpenAIEmbeddings(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="bge-m3",
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    # 需要数据持久化时使用以下参数
    persist_directory="./chroma_langchain_db",
)

ids = vector_store.add_documents(documents=all_splits)


@tool
def search_handbook(query: str) -> str:
    """Search the instruction for use for information."""
    results = vector_store.similarity_search(query, k=8)
    return "\\n".join([doc.page_content for doc in results])


model = init_chat_model(
    model_provider="openai",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.3,
)

agent = create_agent(
    model=model,
    tools=[search_handbook],
    system_prompt="You are a helpful agent that can search the instruction for use for information.",
)

response = agent.invoke({"messages": [HumanMessage(content="运动标靶 清洁 消毒 方法")]})

rprint(response)
