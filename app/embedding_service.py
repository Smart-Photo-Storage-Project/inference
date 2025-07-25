
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO

img_model = SentenceTransformer("clip-ViT-B-32")
text_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

def get_image_embedding(file_bytes: bytes):
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    return img_model.encode(image)

def get_text_embedding(text: str):
    return text_model.encode(text)
