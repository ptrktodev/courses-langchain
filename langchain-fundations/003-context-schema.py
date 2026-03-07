from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from rich.pretty import pprint
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.5)

''''@dataclass # gera automaticamente o __init__ 
class ColourFavorite:
    colour: str = 'red'''

@dataclass # gera automaticamente o __init__ 
class ColourFavorite:
    colour: str 

@tool
def get_favorite_colour(runtime: ToolRuntime) -> str:
    """Returns the user's favorite colour."""
    return runtime.context.colour

agent = create_agent(
    model=llm,
    tools=[get_favorite_colour],
    context_schema=ColourFavorite
)

'''response_ = agent.invoke(
    {"messages": [HumanMessage('What is my favorite colour?')]},
    context=ColourFavorite()
)'''

response_ = agent.invoke(
    {"messages": [HumanMessage('What is my favorite colour?')]},
    context=ColourFavorite(colour='yellow')
)

pprint(response_)
