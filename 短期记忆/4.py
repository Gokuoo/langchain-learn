from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.7,
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_BASE")
    # openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    # openai_api_base=os.getenv("ZHIPUAI_API_BASE"),
)


class CustomState(AgentState):
    user_id: str


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """look up user info."""
    user_id = runtime.state["user_id"]
    return "user is John Smith" if user_id == "user_123" else "Unknown user"


agent = create_agent(
    model=model,
    tools=[get_user_info],
    state_schema=CustomState
)

respone = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})

print(respone["messages"][-1])
print("----------------")
print(respone["messages"][-1].content)
