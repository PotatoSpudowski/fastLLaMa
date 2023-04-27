from fastLLaMa import Model

MODEL_PATH = "./models/VICUNA-7B/ggml-vicuna-7b-1.0-uncensored-q4_2.bin"

def stream_token(x: str) -> None:
    """
    This function is called by the llama library to stream tokens
    """
    print(x, end='', flush=True)

model = Model(
        path=MODEL_PATH, #path to model
        num_threads=16, #number of threads to use
        n_ctx=500, #context size of model
        last_n_size=16, #size of last n tokens (used for repetition penalty) (Optional)
        n_batch=128,
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
        num_tokens=500, 
        top_p=0.95, #top p sampling (Optional)
        temp=0.8, #temperature (Optional)
        repeat_penalty=1.0, #repetition penalty (Optional)
        streaming_fn=stream_token, #streaming function
        stop_words=["###"] #stop generation when this word is encountered (Optional)
        )

    print("\n")
