from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# 使用 ChatOpenAI 接入智谱 AI
llm = ChatOpenAI(
    temperature=0.7,
    model="glm-4.6v",  # 推荐使用官方标准模型名，如 glm-4-plus, glm-4-flash
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),  # 读取你的智谱 API Key
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

messages = [
    SystemMessage(content="你是一个有用的 AI 助手"),
    HumanMessage(content="说一句你好")
]

response = llm.invoke(messages)
print(response)
print(response.content)
