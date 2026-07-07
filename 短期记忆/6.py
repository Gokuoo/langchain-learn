from langchain.agents.middleware import dynamic_prompt, ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import AnyMessage, SystemMessage
from langchain.agents import create_agent, AgentState
from typing import TypedDict, Callable
from langchain.tools import tool
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.7,
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_BASE")
)


class CustomContext(TypedDict):
    user_name: str | None
    user_id: str | None

# @tool


def get_weather(city: str) -> str:
    """get the weather in a city."""
    return f"The weather in {city} is always sunny!"


# @dynamic_prompt
# def dynamic_system_prompt(request: ModelRequest) -> str:
#     user_name = request.runtime.context["user_name"]
#     system_prompt = f"You are a helpful assistant. Address the user as {user_name}"
#     return system_prompt


# 使用 @wrap_model_call 替代 @dynamic_prompt
@wrap_model_call
def dynamic_system_prompt(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    # 1. 从 runtime.context 中获取用户名
    user_name = request.runtime.context.get("user_name", "PPPPPPPPPP")
    # 2. 构造新的系统提示
    new_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    # 3. 创建一个新的 SystemMessage
    new_system_message = SystemMessage(content=new_prompt)
    # 4. 使用 override 方法替换 request 中的系统消息，并继续调用模型
    return handler(request.override(system_message=new_system_message))


agent = create_agent(
    model,
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

respone = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in SF?"}]},
    context=CustomContext(user_id="ooooo"),
)

for msg in respone["messages"]:
    print(msg)
