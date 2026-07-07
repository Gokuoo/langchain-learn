from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()


model = ChatOpenAI(
    model="glm-4.5-air",
    temperature=0.7,
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_BASE")
)

agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=4000,
            messages_to_keep=20,
            summary_prompt="Custom prompt for summarization...",
        ),
    ],
)
