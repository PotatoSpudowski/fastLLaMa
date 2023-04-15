from build.fastllama import Model, ModelKind

MODEL_PATH = "./models/7B/ggml-model-q4_0.bin"

model = Model(
        id=ModelKind.ALPACA_LORA_7B,
        path=MODEL_PATH, #path to model
        num_threads=16, #number of threads to use
        n_ctx=512, #context size of model
        last_n_size=16, #size of last n tokens (used for repetition penalty) (Optional)
        n_batch=512, #number of batches to use
    )

# Perplexity caculation on a file named test.txt (8000 bytes)
with open("test.txt", "r") as f:
    data = f.read(8000)
    total_perplexity = model.perplexity(data)
    print(f"Total Perplexity: {total_perplexity:.4f}")
