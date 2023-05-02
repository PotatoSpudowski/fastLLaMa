from websockets.sync.server import serve
from threading import Lock
import threading
from os import listdir
from os.path import isfile, join
import queue
from fastllama import Model
# from examples.python.build.fastllama import Model
lock = Lock()
message_queue = queue.Queue()
response_queue = queue.Queue()


def stream_token(x: str) -> None:
    global socket
    with lock:
        socket.send(f"ST:{x}")


def update_progress(a, b):
    global socket
    with lock:
        socket.send(f"Prog:{int(255*a/b)}")


MODELS_FOLDER = "./models"
MODEL_PATH = "./models/ALPACA-LORA-7B/alpaca-lora-q4_0.bin"

model = None


def load_model():
    global model, MODEL_PATH
    model = Model(
        path=MODEL_PATH,  # path to model
        num_threads=8,  # number of threads to use
        n_ctx=2048,  # context size of model
        # size of last n tokens (used for repetition penalty) (Optional)
        last_n_size=64,
        seed=0  # seed for random number generator (Optional)
    )


def generate(callback):
    global model
    model.generate(
        num_tokens=512,
        top_p=0.95,  # top p sampling (Optional)
        temp=.8,  # temperature (Optional)
        repeat_penalty=1.2,  # repetition penalty (Optional)
        streaming_fn=callback,  # streaming function
        # stop generation when this word is encountered (Optional)
        stop_words=['###']
    )


def list_models():
    return [f for f in listdir(MODELS_FOLDER) if not isfile(join(MODELS_FOLDER, f))]


def set_model(model_name: str) -> str:
    model_root = f"{MODELS_FOLDER}/{model_name}"
    models = [f for f in listdir(model_root) if isfile(
        join(model_root, f)) and f.endswith("q4_0.bin")]
    return f"{model_root}/{models[0]}"


def echo(websocket: serve):
    global socket, model, message_queue, response_queue, MODEL_PATH
    socket = websocket
    for msg in websocket:
        message = str(msg)

        print(f"recieved: {message}")
        websocket.send(f"Recieved: {message}")
        if message.startswith("P:"):
            prompt = message[2:]
            model.ingest(prompt, update_progress)
            print("prompt ingested !")
            websocket.send("Prog:255")
            generate(stream_token)
        if message == "list_models":
            websocket.send("Models:"+"|".join(list_models()))
        if message.startswith("load_model:"):
            modelname = message[11:]
            newPath = set_model(modelname)
            if(model == None or newPath != MODEL_PATH):
                MODEL_PATH = newPath
                with lock:
                    message_queue.put("load")
                while True:
                    try:
                        message = response_queue.get(block=False)
                    except queue.Empty:
                        pass
                    else:
                        # call the load function when a "load" message is received
                        if message == "loaded":
                            break
            print("model loaded!")
        if(model != None):
            websocket.send(f"UNLOCK")


def start_server():
    with serve(echo, "localhost", 8765) as server:
        server.serve_forever()


def main():
    global message_queue, response_queue
    # Can't have server running in main thread, because main thread needs to be available
    # for some operations such as constructing the new model
    server_thread = threading.Thread(target=start_server)
    server_thread.start()
    while True:
        try:
            message = message_queue.get(block=False)
        except queue.Empty:
            pass
        else:
            # call the load function when a "load" message is received
            if message == "load":
                load_model()
                response_queue.put("loaded")


if __name__ == "__main__":
    main()
