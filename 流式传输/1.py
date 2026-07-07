from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
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
    """获取给定城市的天气信息"""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model,
    tools=[get_weather],
)
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "大连的天气怎么样?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step:{step}")
        print(f"content:{data['messages'][-1]}")
