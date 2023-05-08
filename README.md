# fastLLaMa

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`fastLLaMa` is an experimental high-performance framework designed to tackle the challenges associated with deploying large language models (LLMs) in production environments. 


It offers a user-friendly Python interface to a C++ library, [llama.cpp](https://github.com/ggerganov/llama.cpp), enabling developers to create custom workflows, implement adaptable logging, and seamlessly switch contexts between sessions. This framework is geared towards enhancing the efficiency of operating LLMs at scale, with ongoing development focused on introducing features such as optimized cold boot times, Int4 support for NVIDIA GPUs, model artifact management, and multiple programming language support.

```
                ___            __    _    _         __ __      
                | | '___  ___ _| |_ | |  | |   ___ |  \  \ ___ 
                | |-<_> |<_-<  | |  | |_ | |_ <_> ||     |<_> |
                |_| <___|/__/  |_|  |___||___|<___||_|_|_|<___|
                                                            
                                                                                        
                                                                           
                                                       .+*+-.                
                                                      -%#--                  
                                                    :=***%*++=.              
                                                   :+=+**####%+              
                                                   ++=+*%#                   
                                                  .*+++==-                   
                  ::--:.                           .**++=::                   
                 #%##*++=......                    =*+==-::                   
                .@@@*@%*==-==-==---:::::------::==*+==--::                   
                 %@@@@+--====+===---=---==+=======+++----:                   
                 .%@@*++*##***+===-=====++++++*++*+====++.                   
                 :@@%*##%@@%#*%#+==++++++=++***==-=+==+=-                    
                  %@%%%%%@%#+=*%*##%%%@###**++++==--==++                     
                  #@%%@%@@##**%@@@%#%%%%**++*++=====-=*-                     
                  -@@@@@@@%*#%@@@@@@@%%%%#+*%#++++++=*+.                     
                   +@@@@@%%*-#@@@@@@@@@@@%%@%**#*#+=-.                       
                    #%%###%:  ..+#%@@@@%%@@@@%#+-                            
                    :***#*-         ...  *@@@%*+:                            
                     =***=               -@%##**.                            
                    :#*++                -@#-:*=.                            
                     =##-                .%*..##                             
                      +*-                 *:  +-                             
                      :+-                :+   =.                             
                       =-.               *+   =-                             
                        :-:-              =--  :::                           
                                                                           

```
---

## Features
- [x] Easy-to-use Python interface that allows developers to build custom workflows.
    - [x] Pip install support.
- [x] Ability to ingest system prompts.
    - [x] System prompts will remain in runtime memory, normal prompts are recycled.
- [x] Customisable logger support.
- [x] Low memory mode support using mmap.
- [x] Quick context switching between sessions.
    - [x] Ability to save and load session states.
- [x] Quick LoRA adapter switching during runtime.
    - [x] During the conversion of LoRA adapters to bin file, we are caching the result of matrix multiplication to avoid expensive caclulation for every context switch.
    - [x] Possible quantization of LoRA adapters with minimal performance degradation. (FP16 supported)
    - [x] Attach and Detach support during runtime.
    - [x] Support to attach and detach adapters for models running using mmap.
- [ ] Cold boot time optimization using multithreading.
    - [x] Improve loading using threads.
    - [ ] Support for `aio_read` for posix.
    - [ ] Experiment with Linux `io_uring`.
- [x] [Web Socket Server](https://github.com/PotatoSpudowski/fastLLaMa/tree/websocket-server).
- [x] [Web UI for chat](https://github.com/PotatoSpudowski/fastLLaMa/tree/webui).  
- [ ] Implement Multimodal models like MiniGPT-4
    - [ ] Implement ViT and Q-Former 
    - [ ] TBD ...
- [ ] Int4 support for NVIDIA GPUs.
- [ ] Model artifact management support.
- [ ] Multiple programming language support.

### Supported Models
- [X] LLaMA 🦙
- [X] Alpaca
- [X] GPT4All
- [X] Chinese LLaMA / Alpaca
- [X] Vigogne (French)
- [X] Vicuna
- [X] Koala
---

## Requirements
1. CMake

    * For Linux: \
    ```sudo apt-get -y install cmake```

    * For OS X: \
    ```brew install cmake```


   * For Windows \
Download cmake-*.exe installer from [Download page](https://cmake.org/download/) and run it.

2. GCC 11 or greater
3. Minimum C++ 17
4. Python 3.x

## Installation 

To install `fastLLaMa` through pip use

```bash
pip install git+https://github.com/PotatoSpudowski/fastLLaMa.git@main
```

## Usage

### Importing the package

To import fastLLaMa just run

```python
from fastllama import Model 
```

### Initializing the Model
```python
MODEL_PATH = "./models/7B/ggml-model-q4_0.bin"

model = Model(
        path=MODEL_PATH, #path to model
        num_threads=8, #number of threads to use
        n_ctx=512, #context size of model
        last_n_size=64, #size of last n tokens (used for repetition penalty) (Optional)
        seed=0, #seed for random number generator (Optional)
        n_batch=128, #batch size (Optional)
        use_mmap=False, #use mmap to load model (Optional)
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

res = model.ingest(prompt, is_system_prompt=True) #ingest model with prompt
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
    stop_words=["User:", "\n"] #stop generation when this word is encountered (Optional)
    )
```
### Loading model using Multithreads 

```python
model = Model(
        path=MODEL_PATH, #path to model
        num_threads=8, #number of threads to use
        n_ctx=512, #context size of model
        last_n_size=64, #size of last n tokens (used for repetition penalty) (Optional)
        seed=0, #seed for random number generator (Optional)
        n_batch=128, #batch size (Optional)
        load_parallel=True
    )
```

### Saving Model State

To cache the session, you can use the `save_state` method.

```python
res = model.save_state("./models/fast_llama.bin")
```

### Loading Model State

To load the session, use the `load_state` method.

```python
res = model.load_state("./models/fast_llama.bin")
```

### Resetting the Model State

To reset the session use the `reset` method.

```python
model.reset()
```
### Attaching LoRA Adapters to Base model during runtime

To attach LoRA Adapter during runtime use the `attach_lora` method.

```python
LORA_ADAPTER_PATH = "./models/ALPACA-7B-ADAPTER/ggml-adapter-model.bin"

model.attach_lora(LORA_ADAPTER_PATH)
```

Note: It is a good idea to reset the state of the model after attaching a LoRA Adapter.

### Detaching LoRA Adapters to Base model during runtime

To detach LoRA Adapter during runtime use the `detach_lora` method.

```python
model.detach_lora()
```

### Calculating perplexity

To caculate the perplexity, use the `perplexity` method.

```python

with open("test.txt", "r") as f:
    data = f.read(8000)
       
total_perplexity = model.perplexity(data)
print(f"Total Perplexity: {total_perplexity:.4f}")
```

### Getting the embeddings of the model

To get the embeddings of the model, use the `get_embeddings` method.

```python
embeddings = model.get_embeddings()
```

### Getting the logits of the model

To get the logits of the model, use the `get_logits` method.

```python
logits = model.get_logits()
```

### Using the logger

```python
from fastLLaMa import Logger

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
```

For more clarity, check the `examples/python/` folder.

### Running LLaMA
```sh
# obtain the original LLaMA model weights and place them in ./models
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# convert the 7B model to ggml FP16 format
# python [PythonFile] [ModelPath] [Floattype] [Vocab Only] [SplitType]
python3 scripts/convert-pth-to-ggml.py models/7B/ 1 0

# quantize the model to 4-bits
./build/src/quantize models/7B/ggml-model-f16.bin models/7B/ggml-model-q4_0.bin 2

# run the inference
#Run the scripts from the root dir of the project for now!
python ./examples/python/example.py
```

### Running Alpaca-LoRA 

```sh
# Before running this command
# You need to provide the HF model paths here
python ./scripts/export-from-huggingface.py
# Alternatively you can just download the ggml models from huggingface directly and run them! 

python3 ./scripts/convert-pth-to-ggml.py models/ALPACA-LORA-7B 1 0

./build/src/quantize models/ALPACA-LORA-7B/ggml-model-f16.bin models/ALPACA-LORA-7B/alpaca-lora-q4_0.bin 2

python ./examples/python/example-alpaca.py
```

### Using LoRA adapters during runtime

```sh
# Download lora adapters and paste them inside models folder
# https://huggingface.co/tloen/alpaca-lora-7b


python scripts/convert-lora-to-ggml.py models/ALPACA-7B-ADAPTER/ -t fp32 
# Change -t to fp16 to use fp16 weights
# Inorder to use LoRA adapters without caching, pass the --no-cache flag
#   - Only supported for fp32 adapter weights

python examples/python/example-lora-adapter.py

# Make sure to set paths correctly for the base model and adapter inside the example
# Commands: 
# load_lora: Attaches the adapter to the base model 
# unload_lora: Deattaches the adapter (Deattach for fp16 is yet to be added!)
# reset: Resets the model state
```

### Running the webUI

To run the [WebSocket Server](https://github.com/PotatoSpudowski/fastLLaMa/tree/websocket-server) and the [WebUI](https://github.com/PotatoSpudowski/fastLLaMa/tree/webui), Follow the instructions on the respective branches.

### Memory/Disk Requirements

As the models are currently fully loaded into memory, you will need adequate disk space to save them
and sufficient RAM to load them. At the moment, memory and disk requirements are the same.

| model size | original size | quantized size (4-bit) |
|-------|---------------|------------------------|
| 7B    | 13 GB         | 3.9 GB                 |
| 13B   | 24 GB         | 7.8 GB                 |
| 30B   | 60 GB         | 19.5 GB                |
| 65B   | 120 GB        | 38.5 GB                |

**Info:** Run time may require extra memory during inference!\
(Depends on hyperparmeters used during model initialization)

### Contributing
* Contributors can open PRs
* Collaborators can push to branches to the repo and merge PRs into the main branch
* Collaborators will be invited based on contributions
* Any help with managing issues and PRs is very appreciated!
* Make sure to read about our [vision](https://github.com/PotatoSpudowski/fastLLaMa/discussions/46)

### Notes

* Tested on
    * Hardware: Apple silicon, Intel, Arm (Pending)
    * OS: MacOs, Linux, Windows (Pending), Android (Pending)