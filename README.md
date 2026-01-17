# nanoGPT4NKI

Compile a nanoGPT-style GPT-2 model to **AWS Trainium (Trn1)** using **PyTorch NeuronX** (`torch-neuronx`).

The model **forward pass** runs on Neuron (Trainium), while the **token-by-token generate loop** stays on CPU (a simple Python loop) and calls the compiled Neuron graph each step.

## Goal
- Minimal pipeline: **trace/compile → save → load → forward on Trn1**.
- Use `torch_neuronx.trace(...)` (a function) to compile a TorchScript module, then `torch.jit.save()` / `torch.jit.load()`.

## Environment (required)
- A Neuron/Trainium machine (e.g. `trn1.*`) with the AWS Neuron SDK installed.
- `torch-neuronx` and `neuronx-cc` (versions must match your Neuron SDK).
- This repo also uses `transformers` (GPT-2 weights), and `tiktoken` (tokenizer).

If you are using the AWS Neuron DLAMI, you may have a prebuilt “NxD inference” PyTorch venv. Example (path may vary):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -c "import torch, torch_neuronx; print(torch.__version__)"
```

## Repo files (current)
- `nanoGPT4NKI/model.py`: GPT model (nanoGPT-style) with `GPT.from_pretrained(...)`.
- `nanoGPT4NKI/run_model.py`: trace+compile to Neuron, save/load TorchScript, and run a CPU-side generate loop.
- `nanoGPT4NKI/sample.py`: baseline sampling on CPU/GPU without Neuron.
- `gpt2_step_neuron.pt`: compiled artifact saved by `run_model.py` (saved in your current working directory). This repo may include an example copy under `nanoGPT4NKI/`.

## How `run_model.py` works (project-specific)
- Wraps the GPT model as `NeuronStep`, which takes `input_ids` with a **fixed shape** `(BATCH, SEQ_LEN)` and returns next-token logits `(BATCH, vocab)`.
- Traces/compiles with:
  - `torch_neuronx.trace(step, example_inputs, compiler_args="--target=trn1")`
  - saves with `torch.jit.save(..., "gpt2_step_neuron.pt")`
  - reloads with `torch.jit.load("gpt2_step_neuron.pt")`
- Runs a CPU-side generation loop that pads/trims to the fixed `SEQ_LEN` and calls the compiled Neuron module each step.

## Quickstart
Option A (recommended): run from the directory **above** `nanoGPT4NKI/`:

```bash
python nanoGPT4NKI/run_model.py
```

Option B: run from inside `nanoGPT4NKI/` (keeps `gpt2_step_neuron.pt` inside that folder):

```bash
cd nanoGPT4NKI
PYTHONPATH=.. python run_model.py
```

It will compile, save+load `gpt2_step_neuron.pt`, run a small CPU-vs-Neuron single-step comparison, and generate text.

## Notes / constraints
- **Static shapes:** tracing compiles for the example input shape (`SEQ_LEN=128`, `BATCH=1` in `nanoGPT4NKI/run_model.py`). If you change shapes, re-trace.
- **Neuron-only artifact:** the saved `.pt` is intended to run on Neuron instances.
- If you already have a saved `.pt`, you can reuse it in your own workflow (skip re-tracing).

## License
See `LICENSE`.
