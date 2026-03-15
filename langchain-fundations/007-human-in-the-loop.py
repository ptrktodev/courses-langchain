from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich.pretty import pprint
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
llm_1 = ChatOpenAI(model="gpt-4-0613", temperature=0.5)

@tool
def sum_numbers(a: int, b: int) -> int:
    """Calcule a soma de dois números."""
    return a + b

@tool
def mult_numbers(a: float, b: float) -> float:
    """Multiplica dois números. Use quando precisar calcular o produto entre valores."""
    return a * b

agent = create_agent(
    model=llm_1,
    tools=[sum_numbers, mult_numbers],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                'mult_numbers': True,
                'sum_numbers': False
            }
        )
    ],
    checkpointer=InMemorySaver()
)

while True:
    input_text = str(input("Digite ('exit' para sair): "))

    if input_text.lower() == "exit":
        break

    response_ = agent.invoke(
        {"messages": [HumanMessage(input_text)]},
        {"configurable": {"thread_id": "1"}},
    )

    if '__interrupt__' in response_:
        response_ = agent.invoke(
            Command( 
                resume={"decisions": [{"type": "approve"}]}
            ), 
            {"configurable": {"thread_id": "1"}},
        )

    pprint(response_)
