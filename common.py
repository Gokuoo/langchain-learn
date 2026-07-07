from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="glm-air-4.5",
    temperature=0.7,
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    openai_api_base=os.getenv("ZHIPUAI_API_BASE")
)
