from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_core.pydantic_v1 import BaseModel
from langchain.prompts import PromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from utils import llm
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

duck_search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="grocery search",
        func=duck_search.run,
        description="""
        Use it to search for the camp food recipe the user asks for in the input.
        Provide the ingredients to make the recipe, and the directions to make the recipe
        """
    )
]


system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['tool_names', 'tools'],
        template='''
        You are a helpful cooking assistant that searches for the a particular camp food recipe, and provides the ingredients and directions for the recipe. 
        
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
        Thought: ALways think about the user query in this way. Given a meal provide only the ingredients needed to prepare the meal.
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
        DIRECTIONS as the second sub-header. Provide the directions for making the recipe as the content of the second sub-header
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

grocery_agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
).with_types(input_type=InputType)

