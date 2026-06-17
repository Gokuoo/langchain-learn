from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call, wrap_tool_call

import os
from dotenv import load_dotenv

load_dotenv()

basic_model = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    temperature=0.7
)

advanced_model = ChatOpenAI(
    model="glm-4.6v",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    temperature=0.7
)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂性选择模型。"""
    message_count = len(request.state["messages"])

    if message_count > 5:
        model = advanced_model
    else:
        model = basic_model

    request = request.override(model=model)
    return handler(request)


agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selection]
)

respone = agent.invoke(
    {"messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
        {"role": "user", "content": "今天天气怎么样"},
        {"role": "assistant", "content": "今天晴天，25度"}
    ]}
)

print(respone)
