from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from dotenv import load_dotenv
import os
import time
import asyncio

load_dotenv()

model = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base=os.getenv("ZHIPUAI_API_BASE"),
    temperature=0.7
)


@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """look up user info."""
    writer = runtime.stream_writer
    store = runtime.store

    # 流式输出进度
    writer(f"🔍 正在查询用户: {user_id}...")
    time.sleep(0.3)

    writer(f"📡 正在访问数据库...")
    time.sleep(0.3)

    user_info = store.get(("users",), user_id)

    if user_info:
        writer(f"✅ 成功找到用户 {user_id}")
        return str(user_info.value)
    else:
        writer(f"❌ 未找到用户 {user_id}")
        return "Unknown user"


@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """save user info."""
    writer = runtime.stream_writer
    store = runtime.store

    # 流式输出保存进度
    writer(f"💾 开始保存用户: {user_id}")
    time.sleep(0.2)

    writer(
        f"📝 验证用户数据: {user_info.get('name', 'N/A')}, {user_info.get('age', 'N/A')}岁")
    time.sleep(0.3)

    store.put(("users",), user_id, user_info)

    writer(f"✅ 用户 {user_id} 保存成功!")
    return f"successfully saved user info for {user_id}"


@tool
def process_batch_users(user_ids: list[str], runtime: ToolRuntime) -> str:
    """批量处理用户信息，演示流式进度更新。"""
    writer = runtime.stream_writer
    store = runtime.store

    total = len(user_ids)
    writer(f"📊 开始批量处理 {total} 个用户...")

    results = []
    for idx, user_id in enumerate(user_ids, 1):
        # 流式输出每个用户的处理进度
        writer(f"⏳ 处理进度: {idx}/{total} - 用户 {user_id}")
        time.sleep(0.2)

        user_info = store.get(("users",), user_id)
        if user_info:
            results.append(
                f"{user_id}: {user_info.value.get('name', 'Unknown')}")
        else:
            results.append(f"{user_id}: 未找到")

        # 每处理2个用户输出一次汇总
        if idx % 2 == 0:
            writer(
                f"📈 已处理 {idx} 个用户，找到 {len([r for r in results if '未找到' not in r])} 个有效用户")

    writer(f"🎉 批量处理完成！共处理 {total} 个用户")
    return "Batch results:\n" + "\n".join(results)


# 初始化存储
store = InMemoryStore()

# 创建 Agent
agent = create_agent(
    model=model,
    tools=[get_user_info, save_user_info, process_batch_users],
    store=store
)


async def stream_agent_response(messages: list[dict]):
    """
    异步流式调用 Agent，实时输出工具执行过程中的流式消息。

    注意：需要监听两种事件：
    1. on_tool_stream - 工具内部通过 writer() 发送的内容
    2. on_chat_model_stream - LLM 的思考过程（可选）
    """
    async for event in agent.astream_events(
        {"messages": messages},
        version="v2"
    ):
        # 捕获工具流式输出
        if event["event"] == "on_tool_stream":
            chunk = event["data"].get("chunk")
            if chunk:
                print(f"  {chunk}", end="", flush=True)

        # 可选：捕获 LLM 思考过程
        elif event["event"] == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                # 不打印 LLM 的内部思考，保持输出清晰
                pass


def run_demo():
    """运行完整演示"""
    print("=" * 60)
    print("🚀 LangChain Stream Writer 完整演示")
    print("=" * 60)

    # ========== 场景1: 保存用户 ==========
    print("\n📌 场景1: 保存用户信息（带流式进度）")
    print("-" * 40)

    # 使用同步调用（不会显示流式输出）
    save_result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "save the following user: userid:abc123, name:张三, age:28, email:zhangsan@qq.com"
            }
        ]
    })
    print(f"✅ 保存结果: {save_result['messages'][-1].content}\n")

    # ========== 场景2: 查询用户（使用流式） ==========
    print("📌 场景2: 查询用户信息（流式输出）")
    print("-" * 40)

    # 使用异步流式调用
    async def query_user():
        await stream_agent_response([
            {"role": "user", "content": "Get user info for user with id 'abc123'"}
        ])
        print("\n")  # 换行

    asyncio.run(query_user())

    # ========== 场景3: 批量处理 ==========
    print("📌 场景3: 批量处理用户（展示复杂进度）")
    print("-" * 40)

    # 先保存几个用户
    test_users = [
        ("user001", {"name": "李四", "age": 25, "email": "lisi@qq.com"}),
        ("user002", {"name": "王五", "age": 32, "email": "wangwu@qq.com"}),
        ("user003", {"name": "赵六", "age": 29, "email": "zhaoliu@qq.com"}),
    ]

    for uid, info in test_users:
        store.put(("users",), uid, info)

    async def batch_process():
        print("📦 已预置 3 个测试用户 (user001, user002, user003)")
        print("开始批量处理...\n")
        await stream_agent_response([
            {"role": "user", "content": "Process batch users: user001, user002, user003, abc123"}
        ])
        print("\n")

    asyncio.run(batch_process())

    # ========== 场景4: 最终查询 ==========
    print("📌 场景4: 最终状态查询")
    print("-" * 40)
    final_response = agent.invoke({
        "messages": [
            {"role": "user", "content": "List all users and their names"}
        ]
    })
    print(f"📋 最终结果:\n{final_response['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("✨ 演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
