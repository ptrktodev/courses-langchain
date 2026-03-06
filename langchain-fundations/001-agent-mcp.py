from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich.pretty import pprint
import os

load_dotenv()

tavily_key = os.environ['TAVILY_API_KEY']
api_key = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model="gpt-4-0613", temperature=0.9)
dir = '/home/patrick/Documents/langchain-courses/langchain-fundations'

# transport: como cliente e servidor se comunicam (stdio = via terminal / sse = HTTP / WebSocket = WebSocket)
# command: comando para iniciar o servidor
# args: argumentos passados pro comando -> npx -y @modelcontextprotocol/server-filesystem /home/patrick/Documents 

# Configs do cliente MCP, se conecta a múltiplos servidores de MCP.
client = MultiServerMCPClient(
    {
        # server que fornece tools para manipular arquivos no sistema.
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                f"{dir}"  # pasta que o servidor pode acessar
            ],
        },
        # mecanismo de busca de informações na internet.
        "tavily": {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server_tavily"],
            "env": {
                "TAVILY_API_KEY": tavily_key
            }
        }
    }
)   

async def main():
    tools = await client.get_tools()

    agent = create_agent( 
        model=llm,
        tools=tools, 
        system_prompt=f"Você tem acesso apenas à pasta {dir}"
    )

    input_user = str(input('Digite sua pergunta: '))
    question = HumanMessage(content=input_user)
    
    response = await agent.ainvoke(
        {"messages": [question]}
    )
    
    pprint(response["messages"])

asyncio.run(main())
