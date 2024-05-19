import os
from dotenv import load_dotenv

from langchain_community.chat_models.fireworks import ChatFireworks
load_dotenv()

fireworks_api_key = os.environ['FIREWORKS_API_KEY']
MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

llm = ChatFireworks(
    model=MODEL_ID,
    model_kwargs={
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 1,
    },
    cache=True,
)

nomic_api_key = os.environ['NOMIC_API_KEY']
