from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv
import nest_asyncio
import os
import time

# permite rodar código async dentro de outro código async que já está em execução 
nest_asyncio.apply()

# tempo atual em segundos com a maior precisão disponível no sistema
start = time.perf_counter()

load_dotenv()

# carregamento de api_keys de .env 
api_key_google = os.environ['GOOGLE_API_KEY']
api_key_qdrant = os.environ['QDRANT_API_KEY']
url_qdrant = os.environ['QDRANT_API_URL']

# definição do modelo default para geração de texto recuperado
Settings.llm = GoogleGenAI(
    api_key=api_key_google,
    model="models/gemini-2.5-flash"
)

# definição do modelo default para embedding
Settings.embed_model = GoogleGenAIEmbedding(
    api_key=api_key_google,
    model_name="models/gemini-embedding-001",
)

# conexão com o qdrant
qdrant_client = QdrantClient(url=url_qdrant, api_key=api_key_qdrant)

collections = qdrant_client.get_collections().collections
names = [c.name for c in collections]
name_new_collection = 'parallelization'

# interface de conexão com o banco vetorial (a ponte)
vector_store_qdrant = QdrantVectorStore(client=qdrant_client, collection_name=name_new_collection)

if name_new_collection not in names:

    # lê o PDF e carrega como lista de Documents (um por página)
    reader = SimpleDirectoryReader(input_files=["paralelizacao.pdf"])
    documents = reader.load_data()

    # divide os documents em chunks em 512 tokens, com 50 tokens de sobreposição entre chunks
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    # criação da collection no cluster do qdrant
    qdrant_client.create_collection(
        collection_name=name_new_collection,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    # definição de onde os dados serão armazenados
    storage_context = StorageContext.from_defaults(vector_store=vector_store_qdrant)

    # embeda e salva no vector store que foi configurado
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    print(f"{len(nodes)} nodes inseridos no Qdrant.")
else:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store_qdrant)
    print("Coleção já existe, carregando do Qdrant.")

# transforma o index em um objeto que você pode fazer perguntas em linguagem natural:
query_engine = index.as_query_engine(similarity_top_k=4)
response = query_engine.query("o que é parallelização?")

print()
print(response)
print(f"\nTempo: {time.perf_counter() - start:.4f}s")