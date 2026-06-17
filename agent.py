from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
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

# 使用 create_agent 构建对话智能体
# 记忆功能通过 messages 列表自然实现
agent = create_agent(
    model=llm,
    system_prompt="You are a nice chatbot having a conversation with a human.",
    # 不传 tools，就是一个纯对话智能体
)

# 手动维护对话历史
chat_history = []


def chat(question: str) -> str:
    """发送消息并获取回复"""
    # 构建消息列表（历史 + 新问题）
    messages = []

    # 添加历史对话
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # 添加当前问题
    messages.append(HumanMessage(content=question))

    # 调用智能体
    response = agent.invoke({"messages": messages})

    # 提取回复内容
    ai_response = response["messages"][-1].content

    # 保存到历史
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": ai_response})

    return ai_response


# 进行对话
print("AI:", chat("给我讲个笑话"))
print("AI:", chat("再来一个"))
