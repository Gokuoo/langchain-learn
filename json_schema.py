import json
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

json_schema = {
    "title": "Movie",
    "description": "一部带有详细信息的电影",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "电影标题"
        },
        "year": {
            "type": "integer",
            "description": "电影上映年份"
        },
        "director": {
            "type": "string",
            "description": "电影导演"
        },
        "rating": {
            "type": "number",
            "description": "电影评分，满分 10 分"
        }
    },
    "required": ["title", "year", "director", "rating"]
}

model = ChatOpenAI(
    model="glm-4.5-air",
    openai_api_base=os.getenv("ZHIPUAI_API_BASE"),
    openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
    temperature=0.7
)

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",
    include_raw=True
)
response = model_with_structure.invoke("提供关于电影《盗梦空间》的详细信息")
print(response)
