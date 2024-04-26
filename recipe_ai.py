from typing import List

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.tools.render import render_text_description
from langchain.tools.retriever import create_retriever_tool 
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from utils import llm

duck_search = DuckDuckGoSearchRun()

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
db_retriever = db.as_retriever()


db_tool_description = (
    "A tool that looks up camp food recipes, it's ingredients, and directions and how to make the food."
    "Use this tool to look up ingredients and directions for making camp food recipes."
)

db_tool = create_retriever_tool(db_retriever, "docstore", db_tool_description)

search_tool = Tool(
        name="plan b",
        func=duck_search.run,
        description="""
        useful for when you can't find a food recipe for the user from {db_retriever}.
        Search for a camp food recipe that contains at least all the ingredients in the user's input. 
        Provide the ingredients to make the recipe, the equipments needed and the directions to make the recipe
        """)

# I just added the search functionality, sometimes it produces an error but most times it doesn't
# I haven't looked into it properly because of limited fireworks credits
# But the error is usually solved by just retrying the query or reformatting the query, like, "I need a recipe for (list the ingredients)"
# Or you can just remove the functionality

tools = [db_tool, search_tool]


system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['tool_names', 'tools'],
        template='''
        You are a helpful cooking assistant that recommends camp food recipes based on ingredients provided by the user. 
        
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
        Thought: ALways think about the user query in this way. Given an ingredient or a list of ingredients follow the following steps: look at and take note of all the provided ingredients, select a recipe by random from {tools} for the user that contains all the ingredients mentioned in the input. You will provide the user with all the ingredients for the recipe, the equipments to be used, and the directions/method to prepare it.
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
        EQUIPMENTS TO USE as the second sub-header. List the equipments to prepare the recipe as the content of the second sub-header
        DIRECTIONS as the third sub-header. Write down the steps to prepare the recipe as the content of the third sub-header
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