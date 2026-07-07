from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


model = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.7,
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_BASE")
)


def get_weather(city: str) -> str:
    """获取给定城市的天气情况。"""
    return f"{city} 总是个晴天！"


agent = create_agent(
    model,
    tools=[get_weather]
)
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "沈阳天气怎么样？"}]},
    stream_mode="messages"
):
    print(f"node:{metadata['langgraph_node']}")
    print(f"content:{token.content_blocks}")
    print("\n")
