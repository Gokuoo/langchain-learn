from langchain.agents import create_agent
# 🎯 关键导入1：智谱AI的官方集成类
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatZhipuAI(
    model="glm-4.6v",          # 智谱AI最新的GLM-4.6模型
    temperature=0.7,          # 控制回答的随机性
)

agent = create_agent(
    model=model
)

messages = [
    SystemMessage(content="你是一个有用的 AI 助手"),
    HumanMessage(content="请介绍一下人工智能的发展历程")
]

# 调用模型
response = agent.invoke({"messages": messages})
print(response.content)
