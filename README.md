# fastLLaMa

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`fastLLaMa` is a high-performance framework designed to tackle the challenges associated with deploying large language models (LLMs) in production environments. 


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
- [x] Ability to ingest system prompts.
    - [x] System prompts will remain in runtime memory, normal prompts are recycled)
- [x] Customisable logger support.
- [x] Quick context switching between sessions.
    - [x] Ability to save and load session states
- [x] Quick LoRA adapter switching during runtime.
    - [x] During the conversion of LoRA adapters to bin file, we are caching the result of matrix multiplication to avoid expensive caclulation for every context switch.
    - [ ] Possible quantization of LoRA adapters with minimal performance degradation (To reduce size of adapters)
- [ ] Int4 support for NVIDIA GPUs.
- [ ] Cold boot time optimization using multithreading.
- [ ] Model artifact management support.
- [ ] Multiple programming language support.

### Supported Models

| model | model_id | status |
|-------|---------------|------------------------|
| LLaMa 7B    | LLAMA-7B | Done |
| LLaMa 13B   | LLAMA-13B | Done |
| LLaMa 30B   | LLAMA-30B | Done |
| LLaMa 65B   | LLAMA-65B | Done |
| Alpaca-LoRA 7B   | ALPACA-LORA-7B | Done |
| Alpaca-LoRA 13B   | ALPACA-LORA-13B | Done |
| Alpaca-LoRA 30B   | ALPACA-LORA-30B | Done |
| Alpaca-LoRA 65B   | ALPACA-LORA-65B | Pending |
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
git clone https://github.com/PotatoSpudowski/fastLLaMa
cd fastLLaMa

# install Python dependencies
pip install -r requirements.txt


python setup.py -l python

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

### Initializing the Model
```python
MODEL_PATH = "./models/7B/ggml-model-q4_0.bin"

model = Model(
        id=ModelKind.LLAMA_7B,
        path=MODEL_PATH, #path to model
        num_threads=8, #number of threads to use
        n_ctx=512, #context size of model
        last_n_size=64, #size of last n tokens (used for repetition penalty) (Optional)
        seed=0, #seed for random number generator (Optional)
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
    stop_word=["User:", "\n"] #stop generation when this word is encountered (Optional)
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

### Running Alpaca-LoRA 

```sh
pip install -r requirements.txt

# Before running this command
# You need to provide the HF model paths here
python ./scripts/export-from-huggingface.py

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

python examples/python/example-lora-adapter.py

# Make sure to set paths correctly for the base model and adapter inside the example
# Commands: 
# load_lora: Attaches the adapter to the base model 
# unload_lora: Deattaches the adapter (Deattach for fp16 is yet to be added!)
# reset: Resets the model state
```

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

<!-- 
### Example Dockerfile
This is a Dockerfile to build a minimal working example. Note that it does not download any models for you. It is also compatible with Alpaca models.

```dockerfile 
FROM ubuntu:18.04
# Add GCC and G++ New
RUN apt-get update && apt-get install software-properties-common curl git -qy 
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA

# Add CMAKE New
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ bionic main" && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4

RUN apt-get update

# Install
RUN apt-get install -qy cmake gcc-10 g++-10

# Configure
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 30 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 30
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30 && update-alternatives --set cc /usr/bin/gcc
RUN update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30 && update-alternatives --set c++ /usr/bin/g++

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y python3.9-dev python3.9-distutils python3.9

WORKDIR /app
RUN git clone https://github.com/PotatoSpudowski/fastLLaMa.git /app
RUN chmod +x build.sh && bash ./build.sh

RUN apt-get install -qy python3-pip
RUN python3.9 -m pip install --upgrade pip && python3.9 -m pip install setuptools-rust
RUN python3.9 -m pip install -r requirements.txt

CMD ["python3.9", "example.py"]
``` -->

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