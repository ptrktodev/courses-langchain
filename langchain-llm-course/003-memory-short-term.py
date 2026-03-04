from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages 
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.9)

def get_history(session_id: int):
    return SQLChatMessageHistory(
        session_id, 
        connection='sqlite:///chat_memory.db'
    )

# Define a estrutura da conversa com o modelo
prompts_template = ChatPromptTemplate.from_messages([
    ('system', "Você é um assistente útil, seja objetivo e direto."),
    MessagesPlaceholder(variable_name="history"),
    ('human', "{user_input}"),
])

# gerencia o tamanho da mensagem de histórico
trimmer = trim_messages(
    max_tokens=10,        # número máximo de mensagens a selecionar
    strategy="last",      # mantém as ultimas mensagens mais recentes
    token_counter=len,    # conta mensagens individualmente (1 por mensagem)
)

# Pipeline de processamento: Template → Modelo → Parser de String
chain = (
    {
        "user_input": lambda x: x["user_input"],
        "history": lambda x: trimmer.invoke(x["history"])  # Aplica o trimmer
    }
    | prompts_template
    | llm
    | StrOutputParser()
)

# Envolve a chain com gerenciamento automático de histórico
runnable_with_history = RunnableWithMessageHistory(
    chain,                          # Chain a ser executada
    get_history,                         # Função para obter histórico
    input_messages_key="user_input",     # Chave do input no template
    history_messages_key="history"       # Chave do histórico no template
)

while True:
    prompt_user = input("Digite seu prompt (ou 'sair' para encerrar): ")
    if prompt_user.lower() == 'sair':
        break
    else:
        # Invoca a chain com histórico
        response = runnable_with_history.invoke(
            {'user_input': prompt_user}, 
            config={
                'configurable': {
                    'session_id': 1  # ID da sessão atual
                }
            }
        )
        
        print(response)
        print() 




        