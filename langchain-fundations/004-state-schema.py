from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command 
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.pretty import pprint
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.5)

class CustomState(AgentState):
    name: str
    age: int

@tool # função que atualiza o state do grafo
def update_info_user(name: str, age: int, runtime: ToolRuntime) -> Command:
    """Update the infos (name and age) of the user in the state once they've revealed it."""
    return Command(update={
        "name": name,
        "age": age,
        "messages": [ToolMessage("Successfully updated name and age", tool_call_id=runtime.tool_call_id)]
        }
    )

@tool
def read_name_and_age(runtime: ToolRuntime) -> str:
    """Read the name and age of the user from the state."""
    try:
        return f"Name: {runtime.state['name']}, Age: {runtime.state['age']}"
    except KeyError:
        return "No name or age found in state"

agent = create_agent(
    model=llm,
    tools=[update_info_user, read_name_and_age],
    state_schema=CustomState,
    checkpointer=InMemorySaver(),
    system_prompt="quando o usuário revelar seu nome e idade, atualize o state usando a função update_info_user. Se o usuário perguntar qual é o nome ou idade dele, use a função read_name_and_age para ler o nome e a idade do state e responder ao usuário."
)

while True:
    input_text = str(input("User: "))
    if input_text.lower() == "exit":
        break
    response_ = agent.invoke(
        {"messages": [HumanMessage(input_text)]},
        {"configurable": {"thread_id": "1"}} #  O checkpointer salva um snapshot completo do estado após cada interação
    )
    pprint(response_)
