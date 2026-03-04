from langchain.tools import tool
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from rich.pretty import pprint
import requests
import os

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.9)

tavily = TavilyClient()
api_weather = os.getenv("WEATHER_API_KEY")

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

tools = [get_capital, get_weather] 

agent = create_agent( 
    model=llm,
    tools=tools, 
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente prestativo."),
    ("human", "{user_input}")
])

chain = prompt | agent

user_input = "qual a temperatura agora em porto alegre/RS?"
response = chain.invoke({"user_input": user_input})

pprint(response)
pprint(f"Resposta final: {response['messages'][-1].content}")
