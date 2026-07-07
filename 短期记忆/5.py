from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.7,
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_BASE"),
)


class CustomState(AgentState):  # [!code highlight]
    user_name: str


class CustomContext(BaseModel):
    user_id: str


@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id  # [!code highlight]
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state["user_name"]
    return f"Hello {user_name}!"
  # [!code highlight]


agent = create_agent(
    model,
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,
)


respone = agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123")
)

print(respone["messages"][-1])
