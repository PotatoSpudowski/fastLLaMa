import sys

sys.path.append("./interfaces/python")

from build.fastllama import Model, ModelKind

MODEL_PATH = "./models/7B/ggml-model-q4_0.bin"

def stream_token(x: str) -> None:
    """
    This function is called by the llama library to stream tokens
    """
    print(x, end='', flush=True)

model = Model(
        id=ModelKind.LLAMA_7B,
        path=MODEL_PATH, #path to model
        num_threads=8, #number of threads to use
        n_ctx=512, #context size of model
        last_n_size=64, #size of last n tokens (used for repetition penalty) (Optional)
        seed=0, #seed for random number generator (Optional)
    )

prompt = """Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Hello, Bob.
Bob: Hello. How may I help you today?
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
User: """

print("\nIngesting model with prompt...")
res = model.ingest(prompt, is_system_prompt=True) #ingest model with prompt

if res != True:
    print("\nFailed to ingest model")
    exit(1)

print("\nModel ingested")

# res = model.save_state("./models/fast_llama.bin") #save model state

# res = model.load_state("./models/fast_llama.bin") #load model state
if not res:
    print("\nFailed to load the model")
    exit(1)
print("\nLoaded the model successfully!")

print("\nGenerating from model...")
print("")

res = model.generate(
    num_tokens=100, 
    top_p=0.95, #top p sampling (Optional)
    temp=0.8, #temperature (Optional)
    repeat_penalty=1.0, #repetition penalty (Optional)
    streaming_fn=stream_token, #streaming function
    # stop_words=[""] #stop generation when this word is encountered (Optional)
    )