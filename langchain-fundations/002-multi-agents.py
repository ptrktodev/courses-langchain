from langchain.tools import tool
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import requests
from rich.pretty import pprint
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.5)

tavily = TavilyClient()
api_weather = os.getenv("WEATHER_API_KEY")

@tool
def sum_numbers(a: float, b: float) -> float:
    """Soma dois números. Use quando precisar adicionar valores."""
    return a + b

@tool
def sub_numbers(a: float, b: float) -> float:
    """Subtrai b de a. Use quando precisar calcular a diferença entre valores."""
    return a - b

@tool
def div_numbers(x: float, y: float) -> object:
    """Divide x por y. Retorna erro se y for zero. Use quando precisar calcular a razão entre dois valores."""
    if y == 0:
        return "Erro: divisão por zero não é permitida."
    return x / y

@tool
def mult_numbers(a: float, b: float) -> float:
    """Multiplica dois números. Use quando precisar calcular o produto entre valores."""
    return a * b

@tool
def get_capital(country: str) -> str:
    """
    Busca a capital de um país usando a API Tavily.

    Args:
        country (str): Nome do país em inglês.

    Returns:
        str: Resultado da busca contendo informações sobre a capital do país.
    """
    capital = tavily.search(
        query=f"What is the capital of {country}?",
        search_depth="ultra-fast",
        max_results=1,
    )
    return capital

@tool
def get_weather(city: str) -> str:
    """
    Obtém as condições climáticas em tempo real de uma cidade via API Tomorrow.io.

    Args:
        city (str): Nome da cidade para consulta meteorológica.

    Returns:
        str | dict: Dados meteorológicos em formato JSON se a requisição for bem-sucedida,
                    ou uma string de erro com o status code em caso de falha.
    """
    url = f"https://api.tomorrow.io/v4/weather/realtime?location={city}&apikey={api_weather}"
    headers = {
        "accept": "application/json",
        "accept-encoding": "deflate, gzip, br"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Erro: status code {response.status_code}"

tools_math = [sum_numbers, mult_numbers, div_numbers, sub_numbers]
tools_utils = [get_capital, get_weather]

agent_math = create_agent(
    model=llm,
    tools=tools_math,
    system_prompt="Você é um agente principal que delega tarefas para dois subagentes: um especializado em operações matemáticas básicas (soma, subtração, multiplicação e divisão) e outro em informações utilitárias (capital de um país e clima atual da capital). Identifique o subagente adequado, delegue a tarefa e retorne a resposta de forma direta e objetiva, sem explicações desnecessárias."
)

agent_utils = create_agent(
    model=llm,
    tools=tools_utils,
    system_prompt="Você é um assistente especializado em fornecer informações sobre capitais de países e condições climáticas em tempo real."
)

@tool
def sub_agent_math(query: str) -> str:
    """Agente especializado em operações matemáticas (soma, subtração, multiplicação e divisão).
    Use quando a tarefa envolver cálculos numéricos.
    
    Args:
        query (str): A consulta contendo a tarefa matemática completa a ser realizada.
    
    """
    response = agent_math.invoke({"messages":[HumanMessage(f"{query}")]})
    return response["messages"][-1].content

@tool
def sub_agent_utils(query: str) -> str:
    """Agente utilitário para tarefas gerais que não envolvam cálculos matemáticos.
    Use quando a tarefa não se encaixar no domínio do agente matemático."""
    response = agent_utils.invoke({"messages":[HumanMessage(f"{query}")]})
    return response["messages"][-1].content

tools_main = [sub_agent_math, sub_agent_utils]

agent_main = create_agent(
    model=llm,
    tools=tools_main,
    system_prompt="Você é o agente principal que delega tarefas para dois subagentes: um especializado em operações matemáticas e outro em informações sobre capitais e clima. Ao receber uma consulta, identifique o subagente adequado, delegue a tarefa e retorne a resposta de forma direta e objetiva."
)

input_user = "Qual é a capital da França e calcula quanto é 15 + 3?"
response_ = agent_main.invoke({"messages": [HumanMessage(input_user)]})
pprint(response_)
