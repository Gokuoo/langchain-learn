from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from langchain.agents.middleware import wrap_tool_call, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage

load_dotenv()


@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"结果：{query}"


@tool
def get_weather(location: str) -> str:
    """获取位置天气信息"""
    return f"{location} 的天气可真棒啊！"


@wrap_tool_call
def handle_tool_erros(request, handler):
    """使用自定义信息处理工具执行错误"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"工具错误:请检查你的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据角色生成系统提示。"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有帮助的助手。"

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术信息"
    elif user_role == "beginner":
        return f"{base_prompt} 提供简单的解释,避免使用行话"

    return base_prompt


model = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_base=os.getenv("ZHIPUAI_API_BASE"),
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    temperature=0.7
)

system_prompt = """
  你是一个幽默的天气播报员，
  你可以使用两个工具：
    - search:搜索结果
    - get_weather:查询特定地点的天气
  如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 search 工具来查找他们的位置。
"""

agent = create_agent(
    model=model,
    tools=[search, get_weather],
    middleware=[handle_tool_erros, user_role_prompt],
    system_prompt=system_prompt
)

respone = agent.invoke(
    {"messages": [{"role": "user", "content": "今日大连天气怎么样"}]},
    context={"user_role": "expert"})

print(respone)
