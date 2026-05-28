from langchain_protocol.protocol import UsageInfo
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

qdrant = QdrantClient(url="http://localhost:6333")

query = input("enter your query: ")

embed_query = SparseTextEmbedding(model_name="Qdrant/bm25")

query_token = list(embed_query.embed([query]))

result= qdrant.query_points(
    collection_name="BM25_search",
    query=models.SparseVector(indices=query_token[0].indices.tolist(), values=query_token[0].values.tolist()),
    using="bm25",
    limit=5,
    
)

for i in result.points:
    print(i.payload)
    print("\n")