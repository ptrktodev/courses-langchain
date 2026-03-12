from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from rich.pretty import pprint
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.5)

summarization = SummarizationMiddleware(
        model=llm, 
        trigger=('tokens', 100),
        keep=('messages', 2),
        summary_prompt="Resuma a conversa até agora em poucas palavras, mantendo as informações mais importantes. A resposta deve ser breve e direta ao ponto."
    )

agent = create_agent(
    model=llm,
    checkpointer=InMemorySaver(),
    middleware=[summarization],
)

while True:
    input_text = str(input("User: "))
    if input_text.lower() == "exit":
        break
    response_ = agent.invoke(
        {"messages": [HumanMessage(input_text)]},
        {"configurable": {"thread_id": "1"}} #  State acumula: O checkpointer salva um snapshot completo do state após cada interação
    )
    pprint(response_)
