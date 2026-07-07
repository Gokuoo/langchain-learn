from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.config import get_stream_writer  # [!code highlight]
load_dotenv()


model = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.7,
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_BASE")
)


def get_weather(city: str) -> str:
    """获取给定城市的天气情况。"""
    writer = get_stream_writer()  # [!code highlight]
    # 流式传输任何任意数据
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


agent = create_agent(
    model,
    tools=[get_weather]
)
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "沈阳天气怎么样？"}]},
    stream_mode="custom"
):
    print(chunk)
    print("\n")
