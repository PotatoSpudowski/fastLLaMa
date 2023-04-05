# fastLLaMa

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Python wrapper to run Inference of [LLaMA](https://arxiv.org/abs/2302.13971) models using C++

This repo was built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp)


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

## Description

`fastLLaMa` is a Python package that provides a Pythonic interface to a C++ library, llama.cpp. It allows you to use the functionality of the C++ library from within Python, without having to write C++ code or deal with low-level C++ APIs.

---

## Features
* Easy-to-use Python interface to the C++ library.
* High performance compared to the original [LLaMA repo](https://github.com/facebookresearch/llama), thanks to the C++ implementation.
* Ability to save and load the state of the model with system prompts.

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
cd fast_llama

python setup.py

# obtain the original LLaMA model weights and place them in ./models
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# install Python dependencies
pip install -r requirements.txt

# convert the 7B model to ggml FP16 format
# python [PythonFile] [ModelPath] [Floattype] [Vocab Only] [SplitType]
python3 scripts/convert-pth-to-ggml.py models/7B/ 1 0 1

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
        id=MODEL_ID
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

#Before running this command
#You need to provide the HF model paths as found in the original script 
python export-alpaca-lora.py

# python [PythonFile] [ModelPath] [Floattype] [SplitType]
# SplitType should be 1 for Alpaca-Lora models exported from HF
python3 scripts/convert-pth-to-ggml.py models/ALPACA-LORA-7B 1 0 0

./build/quantize models/ALPACA-LORA-7B/ggml-model-f16.bin models/ALPACA-LORA-7B/alpaca-lora-q4_0.bin 2

python example-alpaca.py
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
```

### Notes
* Tested on 
    * M1 Pro Mac
    * Intel Mac
    * Ubuntu:18.04 - Python 3.9
* The whole inspiration behind fastLLaMa is to let the community test the capabilities of LLaMA by creating custom workflows using Python. 
* This project was possible because of [llama.cpp](https://github.com/ggerganov/llama.cpp), Do have a look at it as well.