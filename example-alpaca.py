import sys

sys.path.append("./build/")

import fastLlama

import pyttsx3
import speech_recognition as sr

MODEL_PATH = "./models/ALPACA-LORA-7B/alpaca-lora-q4_0.bin"

rec = sr.Recognizer()

response_str = ""

def get_text():
    with sr.Microphone() as source:
        audio_data = rec.record(source, duration=5)
        text = rec.recognize_whisper(audio_data)
    return text

def stream_token(x: str) -> None:
    """
    This function is called by the llama library to stream tokens
    """
    global response_str
    response_str += x
    print(x, end='', flush=True)

model = fastLlama.Model(
        id="ALPACA-LORA-7B",
        path=MODEL_PATH, #path to model
        num_threads=8, #number of threads to use
        n_ctx=512, #context size of model
        last_n_size=64, #size of last n tokens (used for repetition penalty) (Optional)
        seed=0 #seed for random number generator (Optional)
    )

# print("Ingesting model with prompt...")
# model.ingest("Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:")

# print("Model ingested")

# res = model.save_state("./models/fast_llama.bin")

# res = model.load_state("./models/fast_llama.bin")

print("")
print("Start of chat (Speak to chatbot)")
print("")

while True:
    print("I'm listening... (5 seconds)")
    user_input = get_text()
    if user_input == None or user_input == "":
        continue
    print("User: " + user_input)
    # user_input = input("User: ")

    response_str = ""

    if user_input == "exit":
        break

    user_input = "### Instruction:\n\n" + user_input + "\n\n ### Response:\n\n"

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
        stop_word=[".\n", "# ", "Tags:", "<?php", "package org", "Home ›"] #stop generation when this word is encountered (Optional)
        )

    #get the reponse_str
    pyttsx3.speak(response_str)

    print("\n")
