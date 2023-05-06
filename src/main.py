import asyncio
from fastapi import FastAPI, WebSocket
from starlette.responses import HTMLResponse
from fastllama import Model
from typing import List, Optional
from threading import Thread
import json
import uuid
import os
import time

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
                height: 100vh;
                margin: 0;
            }
            h1 {
                margin: 0;
                padding: 20px;
                background-color: #333;
                color: white;
            }
            #conversationList {
                height: 100%;
                width: 200px;
                overflow-y: auto;
                background-color: #ccc;
                position: fixed;
                left: 0;
                top: 0;
                z-index: 1;
            }
            #conversationList li {
                padding: 10px;
                cursor: pointer;
            }
            #conversationList li .delete {
                float: right;
                cursor: pointer;
                color: red;
                font-weight: bold;
            }
            #main {
                display: flex;
                flex-direction: column;
                flex-grow: 1;
                margin-left: 200px;
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
        <div id="conversationList"></div>
        <div id="main">
            <h1>FastLLaMa Chat</h1>
            <ul id="messages"></ul>
            <div id="input-container">
                <input type="text" id="messageInput" placeholder="Type a message">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onopen = function() {
                console.log("Connected!");
                // Request the list of conversations
                ws.send(JSON.stringify({"method": "list"}));
            };
            ws.onmessage = function(event) {
                console.log("Message from server: ", event.data);
                var messages = document.getElementById('messages');
                var message_data = JSON.parse(event.data);

                if (message_data.type == 'conversationList') {
                    var conversationList = document.getElementById('conversationList');
                    for (var i = 0; i < message_data.conversations.length; i++) {
                        var conversation = document.createElement('li');
                        var deleteButton = document.createElement('span');
                        deleteButton.textContent = '×';
                        deleteButton.classList.add('delete');
                        deleteButton.onclick = function() { // Add a click event listener to the delete button
                            deleteConversation(this.parentNode.id);
                        };
                        conversation.textContent = message_data.conversations[i];
                        conversation.id = message_data.conversations[i]; // Set the conversation ID as the element ID
                        conversation.onclick = function() { // Add a click event listener to the conversation element
                            loadConversation(this.id);
                        };
                        conversation.appendChild(deleteButton);
                        conversationList.appendChild(conversation);
                    }
                } else {
                    displayMessage(message_data);
                }
            };
            ws.onclose = function() {
                console.log("Disconnected");
            };

            function sendMessage() {
                var messageInput = document.getElementById('messageInput');
                var message = messageInput.value;

                var message_dict = {"method": "generate", "text": message};
                
                ws.send(JSON.stringify(message_dict));

                var messages = document.getElementById('messages');
                var user_message = document.createElement('li');
                user_message.textContent = message;
                user_message.classList.add('user');
                messages.appendChild(user_message);

                messageInput.value = '';
            }

            function loadConversation(conversationId) {
                // Clear existing messages
                var messages = document.getElementById("messages");
                while (messages.firstChild) {
                    messages.removeChild(messages.firstChild);
                }

                ws.send(JSON.stringify({"method": "reset"}));
                ws.send(JSON.stringify({"method": "load", "conversationId": conversationId}));
            }

            function displayMessage(message_data) {
                var messages = document.getElementById('messages');
                var existing_message = document.getElementById(message_data.id);
                if (existing_message) {
                    existing_message.textContent += message_data.text;
                } else {
                    var message = document.createElement('li');
                    message.id = message_data.id;
                    message.textContent = message_data.text;
                    message.classList.add(message_data.type);
                    messages.appendChild(message);
                }
            }

            function deleteConversation(conversationId) {
                ws.send(JSON.stringify({"method": "delete", "conversationId": conversationId}));
                var conversation = document.getElementById(conversationId);
                conversation.parentNode.removeChild(conversation);
            }
        </script>
    </body>
</html>
"""

clients: List[WebSocket] = []

MODEL_PATH = "../models/VICUNA-7B/ggml-vicuna-7b-1.0-uncensored-q4_2.bin"

model = Model(
    path=MODEL_PATH,
    num_threads=16,
    n_ctx=500,
    last_n_size=16,
    n_batch=128,
    load_parallel=True,
)

@app.get("/")
async def get():
    return HTMLResponse(html)

def generate_text(websocket, conversation_id, input_data):
    user_input = "\n\n### Instruction:\n\n" + input_data + "\n\n### Response:\n\n"
    model.ingest(user_input)

    generated_text = ""
    input_message_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    def append_token(x: str) -> None:
        nonlocal generated_text
        nonlocal websocket
        nonlocal input_data
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

    input_data = {"id": input_message_id, "text": input_data, "type": "user", "timestamp": time.time()}
    message_data = {"id": message_id, "text": generated_text, "type": "model", "timestamp": time.time()}

    data = json.dumps(input_data) + "\n" + json.dumps(message_data) + "\n"

    save_conversation(conversation_id, data)

def save_conversation(id, data):
    if not os.path.exists("chats"):
        os.makedirs("chats")

    with open("chats/" + id + ".json", "a") as f:
        f.write(data)
    
    save_state_path = "chats/" + id + ".bin"
    model.save_state(save_state_path)

def list_conversations():
    if not os.path.exists("chats"):
        return []

    #list all files that end with .json
    return [f for f in os.listdir("chats") if f.endswith(".json")]
    

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    
    conversation_id = str(uuid.uuid4())
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            print("Received data: ", data)

            if data["method"] == "generate":
                thread = Thread(target=generate_text, args=(websocket, conversation_id, data["text"]))
                thread.start()
                thread.join()

            elif data["method"] == "list":
                conversations = list_conversations()
                await websocket.send_text(json.dumps({'type': 'conversationList', 'conversations': conversations}))

            elif data["method"] == "load":
                conversation_id = data["conversationId"].split(".json")[0]
                if os.path.exists("chats/" + conversation_id + ".bin"):
                    model.reset()
                    model.load_state("chats/" + conversation_id + ".bin")

                if os.path.exists("chats/" + conversation_id + ".json"):
                    with open("chats/" + conversation_id + ".json", "r") as f:
                        for line in f:
                            message_data = json.loads(line)
                            await websocket.send_text(json.dumps(message_data))

            elif data["method"] == "reset":
                model.reset()

            elif data["method"] == "delete":
                conversation_id = data["conversationId"].split(".json")[0]
                if os.path.exists("chats/" + conversation_id + ".json"):
                    os.remove("chats/" + conversation_id + ".json")
                if os.path.exists("chats/" + conversation_id + ".bin"):
                    os.remove("chats/" + conversation_id + ".bin")

    except Exception as e:
        print(e)
        clients.remove(websocket)