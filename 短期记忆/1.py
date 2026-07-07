from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.7,
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base=os.getenv("ZHIPUAI_API_BASE"),
)


@before_model
def trim_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """keep only the last few message to fit context window."""
    message = state["messages"]

    if len(message) <= 3:
        return None

    first_msg = message[0]
    recent_messages = message[-3:] if len(message) % 2 == 0 else message[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "message": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


agent = create_agent(
    model,
    middleware=[trim_message],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bobbbbb"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
agent.invoke({"messages": "我先在很开心和你聊天，你的名字是tim"}, config)

respone = agent.invoke({"messages": "what's my name?"}, config)

print(respone)
