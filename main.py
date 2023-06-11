from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

@ app.post("/process-prompt")
async def process_prompt(prompt: str):
    bark_url = "https://api.github.com/repos/username/Bark"
    headers = {"Accept": "application/vnd.github.v3+json"}

    response = requests.post(bark_url, headers=headers, json={"prompt": prompt})
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=500, detail="Failed to process prompt")
