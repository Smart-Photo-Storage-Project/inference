import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv

load_dotenv() 

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "photo_embeddings")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

collection_name = COLLECTION_NAME

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

