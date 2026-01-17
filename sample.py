"""
Sample from an official GPT-2 checkpoint (via HuggingFace transformers).

This is a simplified version of sample.py that removes "resume from out_dir"
checkpoints and dataset-specific meta.pkl encoding logic.
"""

from contextlib import nullcontext

import torch
import tiktoken

from model import GPT

# -----------------------------------------------------------------------------
# model + sampling settings
init_from = "gpt2"  # one of: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
start = "Steven Liu is a Cornell ECE student"  # can also specify a file: "FILE:prompt.txt"
num_samples = 1
max_new_tokens = 100
temperature = 0.8
top_k = 200
seed = 1337

# runtime settings
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if (device.startswith("cuda") and torch.cuda.is_bf16_supported())
    else "float16"
)
compile = False

# allow command-line overrides, e.g.:
# python sample_new.py --init_from=gpt2-xl --start="Hello" --device=cpu --dtype=float32
exec(open("configurator.py").read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if device.startswith("cuda") and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if device.startswith("cuda") else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# tokenizer (GPT-2 BPE)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda ids: enc.decode(ids)

# prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# model
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# generate
with torch.no_grad():
    with ctx:
        for _ in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print("---------------")
