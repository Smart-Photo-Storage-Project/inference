from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)

collection_name = "photo_embeddings"

def init_qdrant():
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE,
            ),
        )
    else:
        print(f"Collection '{collection_name}' already exists.")

