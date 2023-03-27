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
4. Portaudio

    * For Linux: \
    ```sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0```
    ```sudo apt-get install ffmpeg libav-tools```

    * For OS X: \
    ```brew install portaudio```

5. Flac

    * For Linux: \
    ```sudo apt-get install -y flac```

    * For OS X: \
    ```brew install flac```

## Usage

### Running Alpaca-LoRA Voice Chat

```sh
pip install -r requirements.txt

#Before running this command
#You need to provide the HF model paths as found in the original script 
python export-alpaca-lora.py

# python [PythonFile] [ModelPath] [Floattype] [SplitType]
# SplitType should be 1 for Alpaca-Lora models exported from HF
python3 convert-pth-to-ggml.py models/ALPACA-LORA-7B 1 1

./quantize models/ALPACA-LORA-7B/ggml-model-f16.bin models/ALPACA-LORA-7B/alpaca-lora-q4_0.bin 2

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