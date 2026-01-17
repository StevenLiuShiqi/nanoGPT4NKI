import time
import torch
import torch_neuronx
import tiktoken

from nanoGPT4NKI.model import GPT

SEQ_LEN = 128
BATCH = 1
MODEL_TYPE = "gpt2"


class NeuronStep(torch.nn.Module):
    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, input_ids):
        # input_ids: (B, S) fixed shape
        logits, _ = self.gpt(input_ids)      # (B, 1, vocab)
        return logits[:, 0, :]               # (B, vocab)


def pad_or_trim(ids, seq_len, pad_id):
    if len(ids) >= seq_len:
        return ids[-seq_len:]
    return [pad_id] * (seq_len - len(ids)) + ids


@torch.no_grad()
def benchmark_single_step(step_fn, input_ids, n_warmup=5, n_iters=20):
    """
    Measure average latency (ms) of a single forward step.
    """
    # warm-up
    for _ in range(n_warmup):
        _ = step_fn(input_ids)

    start = time.time()
    for _ in range(n_iters):
        _ = step_fn(input_ids)
    end = time.time()

    return (end - start) / n_iters * 1000.0


@torch.no_grad()
def compare_cpu_vs_neuron(cpu_model, neuron_model, input_ids):
    """
    Compare:
      1) single-step latency
      2) logits numerical difference
    """
    # ---------- CPU ----------
    def cpu_step(x):
        logits, _ = cpu_model(x)
        return logits[:, -1, :]

    cpu_time_ms = benchmark_single_step(cpu_step, input_ids)

    # ---------- Neuron ----------
    neuron_time_ms = benchmark_single_step(neuron_model, input_ids)

    # ---------- Numerical diff ----------
    logits_cpu = cpu_step(input_ids)
    logits_neuron = neuron_model(input_ids)

    diff = (logits_cpu - logits_neuron).abs()

    print("\n=== CPU vs Neuron: single-step comparison ===")
    print(f"CPU forward latency    : {cpu_time_ms:.2f} ms")
    print(f"Neuron forward latency : {neuron_time_ms:.2f} ms")
    print(f"Speedup                : {cpu_time_ms / neuron_time_ms:.2f}x")
    print(f"Max abs diff           : {diff.max().item():.6e}")
    print(f"Mean abs diff          : {diff.mean().item():.6e}")
    print("===========================================\n")


def main():
    # 1) load model on CPU
    model = GPT.from_pretrained(MODEL_TYPE, dict(dropout=0.0))
    model.eval()

    step = NeuronStep(model).eval()

    # 2) example input for tracing
    example = torch.zeros((BATCH, SEQ_LEN), dtype=torch.long)

    # 3) trace + compile to Neuron
    neuron_step = torch_neuronx.trace(
        step,
        example,
        compiler_args="--target=trn1"
    )

    torch.jit.save(neuron_step, "gpt2_step_neuron.pt")

    # 4) load traced model
    loaded = torch.jit.load("gpt2_step_neuron.pt")

    # tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda ids: enc.decode(ids)
    pad_id = enc.eot_token

    # prepare input
    prompt = "I believe the meaning of life is"
    ids = encode(prompt)

    window = pad_or_trim(ids, SEQ_LEN, pad_id)
    x = torch.tensor(window, dtype=torch.long).unsqueeze(0)

    # ===== CPU vs Neuron comparison =====
    compare_cpu_vs_neuron(
        cpu_model=model,
        neuron_model=loaded,
        input_ids=x,
    )

    # ===== generation loop (Neuron) =====
    max_new_tokens = 100
    temperature = 0.8
    top_k = 200

    for _ in range(max_new_tokens):
        window = pad_or_trim(ids, SEQ_LEN, pad_id)
        x = torch.tensor(window, dtype=torch.long).unsqueeze(0)

        logits = loaded(x)
        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)

    print(decode(ids))


if __name__ == "__main__":
    main()
