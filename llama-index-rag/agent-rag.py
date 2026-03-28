from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from rich.pretty import pprint
from dotenv import load_dotenv
import time
import os

load_dotenv()
start = time.perf_counter()

api_key_google = os.environ['GOOGLE_API_KEY']
api_key_qdrant = os.environ['QDRANT_API_KEY']
url_qdrant = os.environ['QDRANT_API_URL']

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.4)

# definição do modelo default para embedding implicito
Settings.embed_model = GoogleGenAIEmbedding(
    api_key=api_key_google,
    model_name="models/gemini-embedding-001",
)

qdrant_client = QdrantClient(url=url_qdrant, api_key=api_key_qdrant)

@tool
def recuperar_conteudo_parallezation(query_user: str) -> str:
    """
    Busca e recupera informações relevantes de uma base de conhecimento vetorial.
    Use esta tool sempre que o usuário fizer perguntas sobre o assunto PARALELIZAÇÃO em Agentes de IA.
    Recebe uma pergunta em linguagem natural e retorna os trechos
    mais relevantes encontrados. Prefira esta tool antes de responder com conhecimento próprio.

    Args:
        query_user: Pergunta ou termo de busca em linguagem natural.
    """
    vector_store_qdrant = QdrantVectorStore(client=qdrant_client, collection_name='parallelization') # objeto que representa a conexão com uma coleção específica
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_qdrant) # Cria um índice LlamaIndex apontando para esse vector store.
    retriever = index.as_retriever(similarity_top_k=4) # Cria um retriever a partir do índice, configurado para retornar os 4 chunks mais similares a qualquer query.
    nodes = retriever.retrieve(query_user) # recupera usando o retriever configurado acima
    return "\n\n---\n\n".join([node.text for node in nodes])

agent = create_agent(
    model=llm,
    tools=[recuperar_conteudo_parallezation],
    system_prompt=(
        "Sempre que o usuário fizer perguntas sobre paralelização, use uma única vez a tool `recuperar_conteudo_parallezation` "
        "passando uma string com a pergunta do usuário, ela retorna uma strig com a resposta e voce encaminha para o usuario. Nunca responda sobre esse tema com conhecimento próprio sem antes consultar a tool."
        "\n\n"
        "Para outros assuntos, responda normalmente."
    )
)

response_ = agent.invoke(
    {"messages": [HumanMessage('O que é paralelização?')]},
)

pprint(response_)
print(f"\nTempo: {time.perf_counter() - start:.4f}s")
