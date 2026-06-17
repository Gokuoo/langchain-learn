from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


model = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_base=os.getenv("ZHIPUAI_API_BASE"),
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    temperature=0.7
)

agent = create_agent(
    model,
    response_format=ToolStrategy(ContactInfo)
)

# agent = create_agent(
#     model=llm,
#     response_format=ProviderStrategy(ContactInfo)
# )

result = agent.invoke({
    "messages": [{"role": "user", "content": "从以下内容提取联系信息：John Doe, john@example.com, (555) 123-4567"}]
})


print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
