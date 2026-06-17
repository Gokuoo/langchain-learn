from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.messages import HumanMessage
import os
from dotenv import load_dotenv
import asyncio


load_dotenv()

# # 创建带流式输出的 LLM
# llm = ChatOpenAI(
#     model="glm-4.6v",
#     openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
#     openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
#     streaming=True,
#     callbacks=[StreamingStdOutCallbackHandler()]
# )

# # 发送消息（输出会实时流式显示）
# response = llm.invoke([HumanMessage(content="写一首关于春天的诗")])


llm = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)


async def main():
    async for event in llm.astream_events("你好", version="v1"):
        if event["event"] == "on_chat_model_start":
            print(f"输入：{event['data']['input']}")

        elif event["event"] == "on_chat_model_stream":
            print(f"令牌：{event['data']['chunk'].text}")

        elif event["event"] == "on_chat_model_end":
            print(f"完整消息：{event['data']['output'].text}")

        else:
            pass

# asyncio.run(main())


responses = llm.batch([
    "为什么鹦鹉有五颜六色的羽毛？",
    "飞机是如何飞行的？",
    "什么是量子计算？"
])
for response in responses:
    print(response)
