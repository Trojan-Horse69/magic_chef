from typing import List
from langchain.agents import AgentExecutor, Tool
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
from langchain_community.tools import DuckDuckGoSearchRun
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

duck_search = DuckDuckGoSearchRun()

db_tool_description = (
    "A tool that looks up African food recipes, including their ingredients and directions on how to make the food."
)

db_tool = create_retriever_tool(db_retriever, "docstore", db_tool_description)

search_tool = Tool(
    name="recipe search",
    func=duck_search.run,
    description="Use this tool to search for African food recipes online. Provide the ingredients and directions required to prepare the recipe."
)

tools = [db_tool, search_tool]

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['tool_names', 'tools'],
        template='''
        You are a helpful cooking assistant that recommends African food recipes based on ingredients provided by the user. 

        You have access to the following tools:
        {tools}

        Try using "docstore" first for recipes. If no recipe is found, use "recipe search".

        Use the tools by specifying a JSON blob with an `action` key (the name of the tool) and an `action_input` key (the input to the tool). Example:

        ```
        {{
          "action": "$TOOL_NAME",
          "action_input": {{
            "query": "$INPUT"
          }}
        }}
        ```

        Provide your response using this format:
        Question: The user's question.
        Thought: Reflect on the query, note the ingredients, and decide on the next step.
        Action: 
        ```
        $JSON_BLOB
        ```
        Observation: Result of the action.
        Repeat Thought/Action/Observation as needed until you have an answer or reach a conclusion.

        If no recipe is found, say: "Can't provide a recipe for those ingredients."

        Final Answer: The final answer should have be in this format:
        Name of the Recipe as the header
        A brief introduction to the recipe like, "Efo riro is a vegetable soup from Yoruba land.  Essentially it is cooked by mixing Spinach with spices and an assorted garnish of meat or other delicacies. Serve with Pounded yam, Eba, Fufu or Amala."
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

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

prompt = chat_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
model_with_stop = llm.bind(stop=["\nObservation", "Can't provide a recipe for those ingredients."])
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
    max_iterations=5,  # Limit iterations to prevent infinite loops
).with_types(input_type=InputType)


