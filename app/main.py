# app/main.py
import asyncio
import aio_pika
import os
import json
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from app.models import TextEmbeddingRequest
from app.embedding_service import get_text_embedding, search_similar_images, embed_and_store_images_batch
from app.qdrant import client, collection_name, init_qdrant
from app.api_key import verify_api_key
from typing import List
from PIL import Image
from io import BytesIO

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_qdrant()

@app.on_event("startup")
async def load_models_and_consumer():
    # Start RabbitMQ consumer in background
    asyncio.create_task(start_rabbitmq_consumer())

@app.post("/embed/images")
async def embed_images(
    request: Request,
    files: List[UploadFile] = File(...),
    names: List[str] = Form(...),
    user_id: str = Form(...),
    upload_at: int = Form(...),
    paths: List[str] = Form(...)
):
    if not (len(files) == len(names) == len(paths)):
        raise HTTPException(status_code=400, detail="Mismatch in lengths of files, names, and paths.")

    try:
        file_bytes_batch = [await f.read() for f in files]

        points = embed_and_store_images_batch(
            batch_file_bytes=file_bytes_batch,
            names=names,
            user_id=user_id,
            upload_at=upload_at,
            paths=paths,
            request=request
        )

        return {
            "status": "stored in qdrant",
            "stored_count": len(points),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# @app.post("/embed/text", dependencies=[Depends(verify_api_key)])
@app.post("/embed/text")
async def embed_text(request: Request, payload: TextEmbeddingRequest):
    try:
        embedding = get_text_embedding(payload.text, request)
        results = search_similar_images(embedding.tolist(), user_id=payload.user_id, top_k=12)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


BASE_URL = "http://localhost:8080/uploads/"
async def fetch_image_bytes(url: str) -> bytes:
    filename = url.replace("\\", "/").split("/")[-1]
    url = BASE_URL + filename
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to fetch image from URL {url}: {str(e)}")


async def handle_message(message: aio_pika.IncomingMessage):
    async with message.process():
        try:
            payload = json.loads(message.body)

            user_id = payload["user_id"]
            upload_at = payload["upload_at"]
            photos = payload["photos"]

            paths = [p["path"] for p in photos]
            names = [p["name"] for p in photos]

            # Async fetch all image bytes from URL
            file_bytes_batch = await asyncio.gather(*(fetch_image_bytes(p) for p in paths))

            embed_and_store_images_batch(
                batch_file_bytes=file_bytes_batch,
                names=names,
                user_id=user_id,
                upload_at=upload_at,
                paths=paths,
                request=None 
            )

            print(f"Processed {len(paths)} images")

        except Exception as e:
            print(f"Error processing message: {e}")


async def start_rabbitmq_consumer():
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
    connection = await aio_pika.connect_robust(rabbitmq_url)
    channel = await connection.channel()

    queue_name = "embedding_jobs"
    queue = await channel.declare_queue(queue_name, durable=True)

    await queue.consume(handle_message)
    print("Listening to RabbitMQ queue:", queue_name)
