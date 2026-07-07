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


# 方法名 > 描述
# def get_weather(city: str) -> str:
def get_current_time(city: str) -> str:
    """获取指定时区的当前时间。"""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


agent = create_agent(
    model,
    tools=[get_current_time]
)
for stream_model, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "大连天气怎么样"}]},
    stream_mode=["updates", "custom"]
):
    print(f"stream_mode:{stream_model}")
    print(f"content:{chunk}")
    print("\n")
