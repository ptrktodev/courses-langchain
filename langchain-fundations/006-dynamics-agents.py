from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import ModelRequest, wrap_model_call, ModelResponse, dynamic_prompt 
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable
from rich.pretty import pprint
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
llm_1 = ChatOpenAI(model="gpt-4-0613", temperature=0.5)
llm_2 = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)

@dataclass
class UserProfile:
    name: str
    age: int 
    location: str
    language: str 
    role: str

@dynamic_prompt
def system_prompt_dynamic(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    context = request.runtime.context
    user_language = context.language
    user_name = context.name
    user_age = context.age
    user_location = context.location
    
    if user_language and user_name and user_age and user_location:
        return f"You are a helpful assistant that speaks {user_language} with the user {user_name}, he is {user_age} years old and lives in {user_location}."
    
    return "You are a helpful assistant."

@wrap_model_call
def model_dynamic(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    messages_count = len(request.messages)
    context = request.runtime.context
    user_role = context.role

    if messages_count > 4:
        request = request.override(model=llm_2) # cópia da request com apenas o campo model substituído, preservando todo o resto.

    if user_role == "internal":
        pass
    else:
        tool = [sum_numbers]
        request = request.override(tools=tool)

    return handler(request)

@tool
def sum_numbers(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

@tool
def mult_numbers(a: float, b: float) -> float:
    """Multiplica dois números. Use quando precisar calcular o produto entre valores."""
    return a * b

agent = create_agent(
    model=llm_1,
    tools=[sum_numbers, mult_numbers],
    context_schema=UserProfile, 
    middleware=[system_prompt_dynamic, model_dynamic],
    checkpointer=InMemorySaver()
)

while True:
    input_text = str(input("User: "))

    if input_text.lower() == "exit":
        break

    response_ = agent.invoke(
        {"messages": [HumanMessage(input_text)]},
        {"configurable": {"thread_id": "1"}},
        context=UserProfile(
            name="JUlio",
            age=25,
            location="POrto ALegre", 
            language="Portuguese-br",
            role="external"
        ),
    )

    pprint(response_)
