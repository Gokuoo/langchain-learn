from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 每 10 秒 1 个请求
    check_every_n_seconds=0.1,  # 每 100 毫秒检查是否允许发出请求
    max_bucket_size=10,  # 控制最大突发大小。
)
model = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base=os.getenv("ZHIPUAI_API_BASE"),
    rate_limiter=rate_limiter)
# response = model.invoke("为什么鹦鹉会说话？")
# response = model.stream("为什么鹦鹉会说话？")

# for chunk in response:
#     print(chunk.content, end="", flush=True)  # 逐块打印，不换行


# for chunk in model.stream("为什么鹦鹉有五颜六色的羽毛？"):
#     reasoning_steps = [
#         r for r in chunk.content_blocks if r["type"] == "reasoning"]
#     print(reasoning_steps if reasoning_steps else chunk.text)


response = model.invoke("为什么鹦鹉有五颜六色的羽毛？")
reasoning_steps = [
    b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))
