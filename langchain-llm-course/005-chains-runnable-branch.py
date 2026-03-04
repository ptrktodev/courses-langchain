from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser  
from pydantic import BaseModel, Field 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.9)
load_dotenv()

class Rota(BaseModel):  
    escolha: int = Field(description="IA = 1, Humano = 2")  
    pensamento: str = Field(description="Campo para o pensamento que levou a decisão da rota escolhida")  
    confianca: float = Field(description="Nível de confiança na decisão, entre 0 e 1")

def default(rota):
    return "Erro"

prompt_ask = ChatPromptTemplate.from_messages([ 
    ('system', 'Pergunte ao User se ele gostaria de ser atendido por uma IA ou por um humano'),
    ('user', '{input}'),
])

parser = PydanticOutputParser(pydantic_object=Rota)

prompt_branch = ChatPromptTemplate.from_messages([ 
    ("system", """Você é um roteador inteligente. Analise a resposta e decida para qual encaminhar.

Opções disponíveis:
- IA: para um atendimento com IA
- Ser Humano: para um atendimento com Humano
{instructions}"""),
    ("user", "{input}")
]).partial(instructions=parser.get_format_instructions())

user_branch = RunnableBranch(
    (lambda x: x.escolha == 1, lambda rota: f"Você escolheu ser atendido por uma IA. {rota.pensamento} {rota.confianca}"),
    (lambda x: x.escolha == 2, lambda rota: f"Você escolheu ser atendido por um Humano. {rota.pensamento} {rota.confianca}"),
    lambda rota:  'erro'
)

chain = prompt_ask | llm
chain2 = prompt_branch | llm | parser | user_branch

input_user = input('digite: ')
response = chain.invoke({'input': input_user})
print(response.content)

input_user2 = input('digite: ')
response = chain2.invoke({'input': input_user2})
print(response)