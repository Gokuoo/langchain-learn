import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents.structured_output import ToolStrategy


import os
from dotenv import load_dotenv
load_dotenv()

# 创建 LLM
llm = ChatOpenAI(
    temperature=0.6,
    model="glm-4.6v",  # 智谱 AI 推荐使用 glm-4-flash 或 glm-4-plus
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)


def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"


@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str


@dataclass
class WeatherResponseFormat:
    """代理的响应模式。"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "锦州" if user_id == "1" else "大连"


agent = create_agent(
    model=llm,
    tools=[get_weather, get_user_location, get_weather_for_location],
    system_prompt="""你是一位擅长用双关语表达的专家天气预报员。
          你可以使用两个工具：

          - get_weather_for_location：用于获取特定地点的天气
          - get_user_location：用于获取用户的位置

          如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。""",
    response_format=ToolStrategy(WeatherResponseFormat)  # 显式强制使用工具策略
)

# 运行代理
response = agent.invoke(
    {"messages": [{"role": "user", "content": "天气怎么样"}]},
    context=Context(user_id="1")
)

print(response)

# print(json.dumps(response, default=str, indent=2, ensure_ascii=False))

# print(response["messages"][-1].content)
