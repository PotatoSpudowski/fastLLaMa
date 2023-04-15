from build.fastllama import Model, ModelKind, Logger
from enum import Enum

MODEL_PATH = "./models/ALPACA-LORA-7B/alpaca-lora-q4_0.bin"

def stream_token(x: str) -> None:
    """
    This function is called by the llama library to stream tokens
    """
    print(x, end='', flush=True)

class MyLogger(Logger):
    def __init__(self):
        super().__init__()
        self.file = open("logs.log", "w")

    def log_info(self, func_name: str, message: str) -> None:
        #Modify this to do whatever you want when you see info logs
        print(f"[Info]: Func('{func_name}') {message}", flush=True, end='', file=self.file)
        pass
    
    def log_err(self, func_name: str, message: str) -> None:
        #Modify this to do whatever you want when you see error logs
        print(f"[Error]: Func('{func_name}') {message}", flush=True, end='', file=self.file)
    
    def log_warn(self, func_name: str, message: str) -> None:
        #Modify this to do whatever you want when you see warning logs
        print(f"[Warn]: Func('{func_name}') {message}", flush=True, end='', file=self.file)
        

model = Model(
        id=ModelKind.ALPACA_LORA_7B,
        path=MODEL_PATH, #path to model
        num_threads=16, #number of threads to use
        n_ctx=512, #context size of model
        last_n_size=16, #size of last n tokens (used for repetition penalty) (Optional)
        n_batch=128,
        logger=MyLogger()
    )

print("")
print("Start of chat (type 'exit' to exit)")
print("")

while True:
    user_input = input("User: ")

    if user_input == "exit":
        break

    user_input = "\n\n### Instruction:\n\n" + user_input + "\n\n### Response:\n\n"

    res = model.ingest(user_input)

    if res != True:
        break
    
    print("\n")

    res = model.generate(
        num_tokens=100, 
        top_p=0.95, #top p sampling (Optional)
        temp=0.8, #temperature (Optional)
        repeat_penalty=1.0, #repetition penalty (Optional)
        streaming_fn=stream_token, #streaming function
        stop_words=["###"] #stop generation when this word is encountered (Optional)
        )

    print("\n")
