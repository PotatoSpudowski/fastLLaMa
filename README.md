# fastLLaMa

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Python wrapper to run Inference of [LLaMA](https://arxiv.org/abs/2302.13971) models using C++

This repo was built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp)

<img src="assets/fast_llama.jpg" alt="My Image" width="500" height="500">

---

## Description

`fastLLaMa` is a Python package that provides a Pythonic interface to a C++ library, llama.cpp. It allows you to use the functionality of the C++ library from within Python, without having to write C++ code or deal with low-level C++ APIs.

---

## Features
* Easy-to-use Python interface to the C++ library.
* High performance compared to the original [LLaMA repo](https://github.com/facebookresearch/llama), thanks to the C++ implementation.
* Ability to save and load the state of the model with system prompts.

---

## Requirements
1. CMake

    * For Linux: \
    ```sudo apt-get -y install cmake```

    * For OS X: \
    ```brew install cmake```


   * For Windows \
Download cmake-*.exe installer from [Download page](https://cmake.org/download/) and run it.

2. Minimum C++ 17
3. Python 3.x

## Usage

### Example
```sh
git clone https://github.com/PotatoSpudowski/fast_llama
cd fast_llama

chmod +x build.sh

./build.sh

# obtain the original LLaMA model weights and place them in ./models
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# install Python dependencies
pip install -r requirements.txt

# convert the 7B model to ggml FP16 format
python3 convert-pth-to-ggml.py models/7B/ 1

# quantize the model to 4-bits
python3 quantize.py 7B

# run the inference
python example.py
```

### Importing fastLlama
```python
import sys

sys.path.append("./build/")

import fastLlama
```

### Initializing the Model
```python
MODEL_PATH = "./models/7B/ggml-model-q4_0.bin"

model = fastLlama.Model(
        path=MODEL_PATH, #path to model
        num_threads=8, #number of threads to use
        n_ctx=512, #context size of model
        last_n_size=64, #size of last n tokens (used for repetition penalty) (Optional)
        seed=0 #seed for random number generator (Optional)
    )
```

### Ingesting Prompts
```python
prompt = """Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Hello, Bob.
Bob: Hello. How may I help you today?
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
User: """

res = model.ingest(prompt) 
```
### Generating Output
```python
def stream_token(x: str) -> None:
    """
    This function is called by the library to stream tokens
    """
    print(x, end='', flush=True)

res = model.generate(
    num_tokens=100, 
    top_p=0.95, #top p sampling (Optional)
    temp=0.8, #temperature (Optional)
    repeat_penalty=1.0, #repetition penalty (Optional)
    streaming_fn=stream_token, #streaming function
    stop_word="User:" #stop generation when this word is encountered (Optional)
    )
```

### Saving Model State
```python
res = model.save_state("./models/fast_llama.bin")
```

### Loading Model State
```python
res = model.load_state("./models/fast_llama.bin")
```

### Memory/Disk Requirements

As the models are currently fully loaded into memory, you will need adequate disk space to save them
and sufficient RAM to load them. At the moment, memory and disk requirements are the same.

| model | original size | quantized size (4-bit) |
|-------|---------------|------------------------|
| 7B    | 13 GB         | 3.9 GB                 |
| 13B   | 24 GB         | 7.8 GB                 |
| 30B   | 60 GB         | 19.5 GB                |
| 65B   | 120 GB        | 38.5 GB                |


### Notes
* I have only tested this out on an M1 Macbook, Please feel free to test it out on other devices.
* The whole inspiration behind fast_llama is to let the community test the capabilities of LLaMA by creating custom workflows using Python. 
* This project was possible because of [llama.cpp](https://github.com/ggerganov/llama.cpp), Do have a look at it as well.