from fastapi import FastAPI, WebSocket
from typing import List
from .utils.fastllama_websocket import FastllamaWebsocket
import json

app = FastAPI()

clients: List[FastllamaWebsocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    fastllamaWs = FastllamaWebsocket(websocket)
    if len(clients) > 0:
        fastllamaWs.notify_error("Sorry, the server is busy right now. Please try again later.")
        await websocket.close()
        return
    
    clients.append(fastllamaWs)
    
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            await fastllamaWs.handle_ws_message(data)
            
    except Exception as e:
        print(e)
        clients.remove(fastllamaWs)