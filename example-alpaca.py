import sys

sys.path.append("./build/")

import fastLlama

MODEL_PATH = "./models/ALPACA-LORA-7B/alpaca-lora-q4_0.bin"

def stream_token(x: str) -> None:
    """
    This function is called by the llama library to stream tokens
    """
    print(x, end='', flush=True)

model = fastLlama.Model(
        path=MODEL_PATH, #path to model
        num_threads=8, #number of threads to use
        n_ctx=10000, #context size of model
        last_n_size=64, #size of last n tokens (used for repetition penalty) (Optional)
        seed=0 #seed for random number generator (Optional)
    )

print("Ingesting model with prompt...")
model.ingest("You are an AI assistant that based the user's Instruction you will give an Explaination.")

print("Model ingested")

# res = model.save_state("./models/fast_llama.bin")

# res = model.load_state("./models/fast_llama.bin")

print("")
print("Start of chat (type 'exit' to exit)")
print("")

while True:
    user_input = input("User: ")

    if user_input == "exit":
        break

    user_input = "Instruction: " + user_input + "\n Explaination:"

    res = model.ingest(user_input)

    if res != True:
        break
    
    print("")
    
    res = model.generate(
        num_tokens=300, 
        top_p=0.95, #top p sampling (Optional)
        temp=0.8, #temperature (Optional)
        repeat_penalty=1.0, #repetition penalty (Optional)
        streaming_fn=stream_token, #streaming function
        stop_word=".\n" #stop generation when this word is encountered (Optional)
        )

    print("\n")
