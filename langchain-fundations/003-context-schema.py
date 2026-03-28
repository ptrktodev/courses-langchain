from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from rich.pretty import pprint
from dotenv import load_dotenv
import os

load_dotenv()

api_key_google = os.environ['GOOGLE_API_KEY']
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-lite",
    api_key=api_key_google,
)

''''@dataclass # gera automaticamente o __init__ 
class ColourFavorite:
    colour: str = 'red'''

@dataclass # gera automaticamente o __init__ 
class nameUser:
    name: str 

@tool
def get_name_user(runtime: ToolRuntime) -> str:
    """Returns the user's name."""
    return runtime.context.name

agent = create_agent(
    model=llm,
    tools=[get_name_user],
    context_schema=nameUser
)

'''response_ = agent.invoke(
    {"messages": [HumanMessage('What is my favorite colour?')]},
    context=ColourFavorite()
)'''

response_ = agent.invoke(
    {"messages": [HumanMessage('What is my name??')]},
    context=nameUser(name='Patrick')
)

pprint(response_)
