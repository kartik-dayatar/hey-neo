import ollama
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

qdrant = QdrantClient(url="http://localhost:6333")

def bm25_search(query):
    embed_query = SparseTextEmbedding(model_name="Qdrant/bm25")

    query_token = list(embed_query.embed([query]))

    result= qdrant.query_points(
        collection_name="BM25_search",
        query=models.SparseVector(indices=query_token[0].indices.tolist(), values=query_token[0].values.tolist()),
        using="bm25",
        with_payload=True,
        limit=10,)

    payload = [p.payload for p in result.points]
    return payload


def similarity_search(query):
    query_vector = ollama.embeddings(model="mxbai-embed-large:latest",prompt=query)

    result= qdrant.query_points(
        collection_name="Similarity_search",
        query=query_vector['embedding'],
        limit=3,
        with_payload=True,
        with_vectors=False
    )

    payload = [p.payload for p in result.points]

    return payload

def package_search(query):
    query_vector = ollama.embeddings(model="mxbai-embed-large:latest",prompt=query)

    result= qdrant.query_points(
        collection_name="packages",
        query=query_vector['embedding'],
        limit=10,
        with_payload=True,
        with_vectors=False
    )

    payload = [p.payload for p in result.points]

    return payload