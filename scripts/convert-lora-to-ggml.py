from io import BufferedWriter
import json
import os
import re
import struct
import sys
from typing import Any, Mapping, MutableMapping, Sequence, Tuple
import argparse

import torch

from convert import DATA_TYPE_TO_FTYPE, NUMPY_TYPE_TO_DATA_TYPE, DataType

HF_SUBLAYER_TO_GGML: Mapping[str, str] = {
    "self_attn.q_proj": "attention.wq",
    "self_attn.k_proj": "attention.wk",
    "self_attn.v_proj": "attention.wv",
    "self_attn.o_proj": "attention.wo",
    "mlp.gate_proj": "feed_forward.w1",
    "mlp.down_proj": "feed_forward.w2",
    "mlp.up_proj": "feed_forward.w3",
    "input_layernorm": "attention_norm",
    "post_attention_layernorm": "ffn_norm",
    # "norm": "norm",
    # "embed_tokens": "tok_embeddings",
    # "lm_head": "output",
}


def translate_tensor_name(t: str) -> Tuple[str, str]:
    match = re.match(r".*layers\.(\d+)\.(\w+\.\w+)\.lora_(A|B)\.weight", t)
    if match:
        nn = match.group(1)
        sub_layer = match.group(2)
        lora_type = match.group(3)

        sub_layer_renamed = HF_SUBLAYER_TO_GGML.get(sub_layer)
        if sub_layer_renamed is None:
            print(f"Error: unrecognized sub-layer {sub_layer} in tensor {t}")
            sys.exit(1)

        output_string = (
            f"layers.{nn}.{HF_SUBLAYER_TO_GGML[sub_layer]}.weight.lora"
        )
        return (output_string, lora_type)
    else:
        print(f"Error: unrecognized tensor {t}")
        sys.exit(1)


def write_file_header(fout: BufferedWriter, _params: Mapping[str, Any]) -> None:
    fout.write(b"ggla"[::-1])  # magic (ggml lora)
    fout.write(struct.pack("i", 1))  # file version
    # fout.write(struct.pack("ii", params["r"], params["lora_alpha"]))


def write_tensor_header(
    fout: BufferedWriter, name: str, shape: Sequence[int], data_type: DataType
) -> None:
    sname = bytes(name, 'utf-8')
    fout.write(
        struct.pack(
            "iii",
            len(shape),
            len(sname),
            DATA_TYPE_TO_FTYPE[NUMPY_TYPE_TO_DATA_TYPE[data_type]],
        )
    )
    fout.write(struct.pack("i" * len(shape), *shape[::-1]))
    fout.write(sname)
    fout.seek((fout.tell() + 31) & -32)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="Path must contain HuggingFace PEFT LoRA files 'adapter_config.json' and 'adapter_model.bin'",
    )
    parser.add_argument(
        '-t',
        '--dtype',
        choices=['fp16', 'fp32'],
        default='fp32',
        help='Data type to use for the converted model. Default: %(default)s',
        dest='dtype',
    )
    return parser.parse_args(sys.argv[1:])

def read_params(input_json: str) -> Mapping[str, Any]:
    params: MutableMapping[str, Any] = {}

    with open(input_json, "r") as f:
        params = json.load(f)

    if params["peft_type"] != "LORA":
        print(f"Error: unsupported adapter type {params['peft_type']}, expected LORA")
        sys.exit(1)

    if params["fan_in_fan_out"] == True:
        print("Error: param fan_in_fan_out is not supported")
        sys.exit(1)

    if params["bias"] is not None and params["bias"] != "none":
        print("Error: param bias is not supported")
        sys.exit(1)

    # TODO: these seem to be layers that have been trained but without lora.
    # doesn't seem widely used but eventually should be supported
    if params["modules_to_save"] is not None and len(params["modules_to_save"]) > 0:
        print("Error: param modules_to_save is not supported")
        sys.exit(1)
    return params


def normalize_tensors(model: Any, params: Mapping[str, Any]) -> Mapping[str, Tuple[torch.Tensor, str]]:
    r = float(params["r"])
    lora_alpha = float(params["lora_alpha"])
    scale = lora_alpha / r
    tensor_map: MutableMapping[str, Tuple[torch.Tensor, str]] = {}
    for k, v in model.items():
        if k.endswith("lora_A.weight"):
            if v.dtype != torch.float16 and v.dtype != torch.float32:
                v = v.float()
        else:
            v = v.float()
        (tensor_name, type) = translate_tensor_name(k)

        if tensor_name in tensor_map:
            (old_tensor, old_type) = tensor_map[tensor_name]
            new_tensor = torch.matmul(v, old_tensor) if old_type == 'A' else torch.matmul(old_tensor, v)
            new_tensor = new_tensor * scale
            tensor_map[tensor_name] = (new_tensor, "")
        else:
            tensor_map[tensor_name] = (v, type)
    return tensor_map

def main() -> None:
    args = parse_args()
    input_json = os.path.join(args.path, "adapter_config.json")
    input_model = os.path.join(args.path, "adapter_model.bin")

    output_path = os.path.join(sys.argv[1], "ggml-adapter-model.bin")

    params = read_params(input_json)

    model = torch.load(input_model, map_location="cpu")

    print("Normalizing tensors...")
    tensor_map = normalize_tensors(model, params)
    print("Normalization completed.\nWriting output...")
    
    with open(output_path, "wb") as fout:
        fout.truncate()

        write_file_header(fout, params)
        for tname, (v, ltype) in tensor_map.items():
            if ltype != "":
                continue
            
            if args.dtype == 'fp16':
                t = v.half().numpy()
            else:
                t = v.numpy()
            print(f"{tname} {t.shape} {t.dtype} {t.nbytes/1024/1024:.2f}MB")
            write_tensor_header(fout, tname, t.shape, t.dtype)
            t.tofile(fout)

    print(f"Converted {input_json} and {input_model} to {output_path}")

if __name__ == '__main__':
    main()