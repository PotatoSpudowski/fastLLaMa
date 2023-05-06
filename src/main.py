import asyncio
from fastapi import FastAPI, WebSocket
from starlette.responses import HTMLResponse
from fastllama import Model
from typing import List, Optional
from threading import Thread
import json
import uuid

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastLlama Chat</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                height: 100vh;
                margin: 0;
            }
            h1 {
                margin: 0;
                padding: 20px;
                background-color: #333;
                color: white;
            }
            #messages {
                flex-grow: 1;
                overflow-y: auto;
                padding: 10px;
                background-color: #f0f0f0;
                list-style-type: none;
                margin: 0;
                padding: 0;
            }
            #messages li {
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
                max-width: 50%;
            }
            #messages li.user {
                background-color: #4CAF50;
                color: white;
                margin-left: auto;
            }
            #messages li.model {
                background-color: white;
            }
            #input-container {
                display: flex;
                padding: 20px;
                background-color: #333;
            }
            #messageInput {
                flex-grow: 1;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            button {
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                margin-left: 10px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <h1>FastLLaMa Chat</h1>
        <ul id="messages"></ul>
        <div id="input-container">
            <input type="text" id="messageInput" placeholder="Type a message">
            <button onclick="sendMessage()">Send</button>
        </div>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onopen = function() {
                console.log("Connected!");
            };
            ws.onmessage = function(event) {
                console.log("Message from server: ", event.data);
                var messages = document.getElementById('messages');
                var message_data = JSON.parse(event.data);

                var existing_message = document.getElementById(message_data.id);
                if (existing_message) {
                    existing_message.textContent += message_data.text;
                } else {
                    var message = document.createElement('li');
                    message.id = message_data.id;
                    message.textContent = message_data.text;
                    message.classList.add('model');
                    messages.appendChild(message);
                }
            };
            ws.onclose = function() {
                console.log("Disconnected");
            };

            function sendMessage() {
                var messageInput = document.getElementById('messageInput');
                var message = messageInput.value;
                ws.send(message);

                var messages = document.getElementById('messages');
                var user_message = document.createElement('li');
                user_message.textContent = message;
                user_message.classList.add('user');
                messages.appendChild(user_message);

                messageInput.value = '';
            }
        </script>
    </body>
</html>
"""

clients: List[WebSocket] = []

MODEL_PATH = "./models/VICUNA-7B/ggml-vicuna-7b-1.0-uncensored-q4_2.bin"

model = Model(
    path=MODEL_PATH,
    num_threads=16,
    n_ctx=500,
    last_n_size=16,
    n_batch=128,
)

@app.get("/")
async def get():
    return HTMLResponse(html)

def generate_text(websocket, data):
    user_input = "\n\n### Instruction:\n\n" + data + "\n\n### Response:\n\n"
    model.ingest(user_input)

    generated_text = ""
    message_id = str(uuid.uuid4())

    def append_token(x: str) -> None:
        nonlocal generated_text
        nonlocal websocket
        nonlocal data
        nonlocal message_id
        generated_text += x
        message_data = {"id": message_id, "text": x}
        asyncio.run(websocket.send_text(json.dumps(message_data)))

    model.generate(
        num_tokens=500,
        top_p=0.95,
        temp=0.8,
        repeat_penalty=1.0,
        streaming_fn=append_token,
        stop_words=["###"],
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print("Client connected")
    print(clients)
    try:
        while True:
            data = await websocket.receive_text()
            print("Message from client: ", data)
            thread = Thread(target=generate_text, args=(websocket, data))
            thread.start()

    except:
        clients.remove(websocket)