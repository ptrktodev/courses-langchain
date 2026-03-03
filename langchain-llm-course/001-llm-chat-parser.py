from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.9)

ChatTemplate = ChatPromptTemplate([    
    ('system', 'You are a helpful assistant that helps users find information about movies.'),
    ('human', '{question}')
])

chain = ChatTemplate | llm | StrOutputParser()

print(chain.invoke({'question': 'Escreva em pt-br um texto curto e direto sobre aviões!'}))




