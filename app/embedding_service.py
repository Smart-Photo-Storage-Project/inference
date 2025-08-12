
from fastapi import Request
from PIL import Image
from io import BytesIO
from app.qdrant import client, collection_name
import uuid
from qdrant_client.models import PointStruct
from typing import List
from sentence_transformers import SentenceTransformer

img_model = SentenceTransformer("clip-ViT-B-32")
text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

def get_batch_image_embeddings(batch_file_bytes: List[bytes], request: Request):
    images = [Image.open(BytesIO(b)).convert("RGB") for b in batch_file_bytes]
    return img_model.encode(images)

def get_text_embedding(text: str, request: Request):
    return text_model.encode(text)

def embed_and_store_images_batch(
    batch_file_bytes: List[bytes],
    names: List[str],
    user_id: str,
    upload_at: int,
    paths: List[str],
    request: Request
):
    embeddings = get_batch_image_embeddings(batch_file_bytes, request)
    points = []

    for i in range(len(batch_file_bytes)):
        point_id = str(uuid.uuid4())
        payload = {
            "name": names[i],
            "user_id": user_id,
            "upload_at": upload_at,
            "path": paths[i]
        }

        point = PointStruct(
            id=point_id,
            vector=embeddings[i].tolist(),
            payload=payload
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)

    return points


def search_similar_images(embedding: list[float], user_id: str, top_k: int = 5):
    search_result = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=top_k,
        with_payload=True,
        query_filter={
            "must": [
                {
                    "key": "user_id",
                    "match": {"value": user_id}
                }
            ]
        }
    )

    results = []
    for hit in search_result:
        payload = hit.payload
        results.append({
            "id": str(hit.id),
            "name": payload.get("name"),
            "user_id": payload.get("user_id"),
            "upload_at": payload.get("upload_at"),
            "path": payload.get("path"),
            "score": hit.score,
        })
    return results
