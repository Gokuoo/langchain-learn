from langchain.tools import tool
from langchain_openai import ChatOpenAI


import os
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)


@tool
def get_weather(location: str) -> str:
    """获取某个位置的天气。"""
    return f"{location} 天气晴朗。"


# model_with_tools = model.bind_tools([get_weather])  # [!code highlight]

# response = model_with_tools.invoke("波士顿的天气怎么样？")
# for tool_call in response.tool_calls:
#     # 查看模型发出的工具调用
#     print(f"工具：{tool_call['name']}")
#     print(f"参数：{tool_call['args']}")

# 将（可能多个）工具绑定到模型
model_with_tools = model.bind_tools([get_weather])

# 步骤 1：模型生成工具调用
messages = [{"role": "user", "content": "波士顿的天气怎么样？"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)


print(ai_msg)


# 步骤 2：执行工具并收集结果
for tool_call in ai_msg.tool_calls:
    # 使用生成的参数执行工具
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# 步骤 3：将结果传递回模型以获取最终响应
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "波士顿当前天气为 72°F，晴朗。"
