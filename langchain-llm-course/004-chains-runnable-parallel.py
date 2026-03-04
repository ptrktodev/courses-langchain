from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.9)

summary_feel = ChatPromptTemplate.from_messages([
    ('system', "Defina em uma palavra o sentimento da frase."),
    ('human', "{user_input}"),
])

translate_en = ChatPromptTemplate.from_messages([
    ('system', "Converta a linguagem da frase de pt_br para en_us."),
    ('human', "{user_input}"),
])

chain = RunnableParallel({
    'summary': summary_feel | llm,
    'translate': translate_en | llm
})

input = 'Antes amar quem só o mal me deseja a quem, fingindo o bem, só o mal me enseja.'

response = chain.invoke({
    'user_input': input
})

print("Sentimento:", response['summary'].content)
print("Tradução:", response['translate'].content)


