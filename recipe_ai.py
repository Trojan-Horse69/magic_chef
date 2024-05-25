from typing import List
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.tools.retriever import create_retriever_tool 
from langchain_core.pydantic_v1 import BaseModel
from langchain.prompts import PromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
from utils import llm
from pydantic import BaseModel

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

embeddings = hf
db = Chroma(embedding_function=embeddings, persist_directory="./african_recipes_chroma_db")
db_retriever = db.as_retriever()


db_tool_description = (
    "A tool that looks up African food recipes, it's ingredients, and directions on how to make the food."
    "Use this tool to look up ingredients and directions for making African food recipes."
)

db_tool = create_retriever_tool(db_retriever, "docstore", db_tool_description)

tools = [db_tool]


system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['tool_names', 'tools'],
        template='''
        You are a helpful cooking assistant that recommends African food recipes based on ingredients provided by the user. 
        
        You have access to the following tools:
        {tools}

        The way you use the tools is by specifying a JSON blob. Specifically, this JSON should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

        The only values that should be in the "action" field are: {tool_names}

        The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

        ```
        {{
          "action": $TOOL_NAME,
          "action_input": $INPUT
        }}
        ```

        ALWAYS use the following format:
        Question: the input question you must answer
        Thought: ALways think about the user query in this way. Given an ingredient or a list of ingredients follow the following steps: look at and take note of all the provided ingredients, select a recipe by random from {tools} for the user that contains all the ingredients mentioned in the input. You will provide the user with all the ingredients for the recipe, and the directions/method to prepare it.
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: the result of the action
        ... (this Thought/Action/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: The final answer should follow this format:
        Name of the Recipe as the header
        INGREDIENTS as the first sub-header. List the ingredients as the content of the sub-header
        DIRECTIONS as the second sub-header. Write down the steps to prepare the recipe as the content of the third sub-header
        '''
    )
)


human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['agent_scratchpad', 'input'],
        template='{input}\n\n{agent_scratchpad}'
    )
)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, human_prompt]
)

prompt = chat_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
model_with_stop = llm.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | model_with_stop
    | ReActJsonSingleInputOutputParser()
)


class InputType(BaseModel):
    input: str


# instantiate AgentExecutor
recipe_agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
).with_types(input_type=InputType)

