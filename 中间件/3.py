from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain.agents import create_agent


LONG_PROMPT = """
Please be a helpful assistant.

<Lots more context ...>
"""

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-latest"),
    system_prompt=LONG_PROMPT,
    middleware=[AnthropicPromptCachingMiddleware(ttl="5m")],
)

# 缓存存储
agent.invoke({"messages": [HumanMessage("Hi, my name is Bob")]})

# 缓存命中，系统提示词被缓存
agent.invoke({"messages": [HumanMessage("What's my name?")]})
