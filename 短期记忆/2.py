from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.7,
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base=os.getenv("ZHIPUAI_API_BASE"),
)


@after_model
def delete_old_message(state: AgentState, runtime: Runtime) -> dict | None:
    """remove old messages to keep conversation manageable."""
    message = state["messages"]
    if len(message) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in message[:2]]}
    return None


agent = create_agent(
    model,
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_message],
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

for event in agent.stream(
    {"messages": [{"role": "user", "content": "hi!I'm bobxxxx"}]},
    config,
    stream_mode="values"
):
    print([(message.type, message.content) for message in event["messages"]])


for event in agent.stream(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    stream_mode="values"
):
    print([(message.type, message.content) for message in event["messages"]])
