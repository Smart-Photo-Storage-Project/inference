import os
from dotenv import load_dotenv
from fastapi import HTTPException, Header

load_dotenv() 

API_KEY = os.getenv("API_KEY")

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")