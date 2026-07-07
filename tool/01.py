from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base=os.getenv("ZHIPUAI_API_BASE"),
    temperature=0.7
)


@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "successfully saved user info."


store = InMemoryStore()
agent = create_agent(
    model=model,
    tools=[get_user_info, save_user_info],
    store=store
)

agent.invoke({
    "messages": [{"role": "user", "content": "save the following user: userid:abc123, name:foooooo, age:35, email: xxxxx@qq.com"}]
})

respone = agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})

print(respone)
