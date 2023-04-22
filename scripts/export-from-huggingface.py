#Modified version of export_state_dict_checkpoint.py from https://github.com/tloen/alpaca-lora
#Provide the path to the model below as found in the export_state_dict_checkpoint.py

import os
import json
import threading
import time
import argparse

import torch
from peft import PeftModel, LoraConfig

import transformers

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM

parser = argparse.ArgumentParser(description="Load and save the ALPACA-LORA model.")
parser.add_argument("--weights", type=str, choices=["7B", "13B", "30B"], required=True, help="Select the weights for the model (7B, 13B, or 30B).")
parser.add_argument("--base", type=str, required=True)
parser.add_argument("--lora", type=str, required=True)
args = parser.parse_args()

params_map = {
    "7B": {
        "dim": 4096,
        "multiple_of": 256,
        "n_heads": 32,
        "n_layers": 32,
        "norm_eps": 1e-06,
        "vocab_size": -1,
    },
    "13B": {
        "dim": 5120,
        "multiple_of": 256,
        "n_heads": 40,
        "n_layers": 40,
        "norm_eps": 1e-06,
        "vocab_size": -1,
    },
    "30B": {
        "dim": 6656,
        "multiple_of": 256,
        "n_heads": 52,
        "n_layers": 60,
        "norm_eps": 1e-06,
        "vocab_size": -1,
    },
}

def monitor_file_size(file_path, stop_event):
    while not stop_event.is_set():
        file_size = os.path.getsize(file_path)
        file_size_gb = file_size / (1024 ** 3)
        print(f"File size: {file_size_gb:.2f} GB")
        time.sleep(5)  # Check the file size every 5 seconds

stop_event = threading.Event()

params = params_map[args.weights]
base_model = args.base
lora_adapter = args.lora

tokenizer = LlamaTokenizer.from_pretrained(base_model)

base_model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

lora_model = PeftModel.from_pretrained(
    base_model,
    lora_adapter,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

# merge weights
for layer in lora_model.base_model.model.model.layers:
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True

lora_model.train(False)

lora_model_sd = lora_model.state_dict()

n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
dims_per_head = dim // n_heads
base = 10000.0
inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))


def permute(w):
    return (
        w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)
    )


def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)
    )


def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError

new_state_dict = {}
for k, v in lora_model_sd.items():
    new_k = translate_state_dict_key(k)
    if new_k is not None:
        if "wq" in new_k or "wk" in new_k:
            new_state_dict[new_k] = unpermute(v)
        else:
            new_state_dict[new_k] = v

def save_model(new_state_dict, params, save_path):
    os.makedirs(save_path, exist_ok=True)
    torch.save(new_state_dict, f"{save_path}/consolidated.00.pth")

    with open(f"{save_path}/params.json", "w") as f:
        json.dump(params, f)

save_path = f"models/ALPACA-LORA-{args.weights}"
os.makedirs(f"models/ALPACA-LORA-{args.weights}", exist_ok=True)

print("Saving model...")
monitor_thread = threading.Thread(target=monitor_file_size, args=(f"{save_path}/consolidated.00.pth", stop_event))
monitor_thread.start()

torch.save(new_state_dict, f"models/ALPACA-LORA-{args.weights}/consolidated.00.pth")
stop_event.set()
monitor_thread.join()
with open(f"models/ALPACA-LORA-{args.weights}/params.json", "w") as f:
    json.dump(params, f)