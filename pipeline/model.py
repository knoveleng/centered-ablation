"""Everything that touches model weights.

* `capture_last` / `capture_tokens`: last-block output states via Hugging Face (extract stage).
* `load_llm` / `generate`: vLLM generation with the operator hooked on the last block (steer).
* `Head`: the unembedding W_U and final-norm gain G only, for the closed form (diagnose).

HF's `hidden_states[-1]` already has the final norm applied, so states are read with a forward
hook on the last decoder block. vLLM blocks return (hidden, residual) with h = hidden + residual;
the hook evaluates the operator on h and subtracts from `hidden`, so the final norm sees the
steered state exactly.
"""

import json

import torch


def chat(tokenizer, prompt: str, response: str | None = None, **kw) -> str | tuple[str, str]:
    msgs = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    if response is None:
        return prefix
    full = tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": response}], tokenize=False)
    return prefix, full


# ------------------------------------------------------------------ Hugging Face
def load_hf(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def _hook_last(model, store):
    block = model.model.layers[-1]
    return block.register_forward_hook(lambda m, i, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o))


@torch.no_grad()
def capture_last(model, tok, prompts: list[str], batch_size: int = 8) -> torch.Tensor:
    """Last-token state of each chat-formatted prompt. Like vLLM on text prompts, the tokenizer
    adds BOS even when the template has one, so extraction and generation see the same tokens."""
    store, out = {}, []
    handle = _hook_last(model, store)
    try:
        texts = [chat(tok, p) for p in prompts]
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
            model(**enc)
            out.append(store.pop("h")[:, -1].float().cpu())
    finally:
        handle.remove()
    return torch.cat(out)


@torch.no_grad()
def capture_tokens(model, tok, items: list[dict], skip: int, keep: int, max_len: int = 1024):
    """States over the assistant-turn tokens of each (prompt, response) item: per item the mean
    state and `keep` randomly sampled token states (after skipping the first `skip`)."""
    store, means, toks = {}, [], []
    handle = _hook_last(model, store)
    try:
        for it in items:
            prefix, full = chat(tok, it["prompt"], it["response"])
            n_pre = len(tok(prefix, add_special_tokens=False).input_ids)
            ids = tok(full, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_len].to(model.device)
            model(ids)
            H = store.pop("h")[0, n_pre + skip:].float().cpu()
            if len(H) < keep:
                continue
            means.append(H.mean(0))
            idx = torch.randperm(len(H), generator=torch.Generator().manual_seed(len(means)))[:keep]
            toks.append(H[idx])
    finally:
        handle.remove()
    return torch.stack(means), torch.cat(toks).to(torch.bfloat16)


# ------------------------------------------------------------------ vLLM
def load_llm(cfg: dict):
    from vllm import LLM

    m = cfg["model"]
    extra = {"attention_config": {"backend": m["attention_backend"]}} if m.get("attention_backend") else {}
    return LLM(model=m["model_id"], dtype="bfloat16", max_model_len=m["max_model_len"],
               gpu_memory_utilization=m["gpu_memory"],
               enforce_eager=True,           # forward hooks only fire in eager mode
               enable_prefix_caching=False,  # a cached unsteered prefix would skip the hook
               **extra)


def _install(steer, s: float):
    def install(model):
        layers = [mod for name, mod in model.named_modules() if name.endswith("layers")][0]
        block = layers[len(layers) - 1]
        st = steer.to(next(block.parameters()).device)

        def hook(module, inputs, output):
            hidden, residual = output
            h = (hidden + residual).float()
            return hidden - (h - st.apply(h, s)).to(hidden.dtype), residual

        block._forward_hooks.clear()
        block.register_forward_hook(hook)
        return len(block._forward_hooks)
    return install


def _clear(model):
    layers = [mod for name, mod in model.named_modules() if name.endswith("layers")][0]
    layers[len(layers) - 1]._forward_hooks.clear()
    return True


def generate(llm, prompts: list[str], gen: dict, steer=None, s: float = 0.0) -> list[str]:
    """Greedy (by default) completions; the operator is installed only when s != 0."""
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    texts = [chat(tok, p) for p in prompts]
    sp = SamplingParams(temperature=gen["temperature"], max_tokens=gen["max_tokens"])
    if steer is None or s == 0:
        outs = llm.generate(texts, sp, use_tqdm=False)
    else:
        assert llm.apply_model(_install(steer, s)) == [1]
        try:
            outs = llm.generate(texts, sp, use_tqdm=False)
        finally:
            llm.apply_model(_clear)
    return [o.outputs[0].text.strip() for o in outs]


# ------------------------------------------------------------------ unembedding head
class Head:
    """W_U and G read straight from the checkpoint (no model load)."""

    def __init__(self, model_id: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        conf = json.load(open(hf_hub_download(model_id, "config.json")))
        tied = bool(conf.get("tie_word_embeddings", False))
        gemma = "gemma" in conf.get("model_type", "").lower()
        unemb = "model.embed_tokens.weight" if tied or gemma else "lm_head.weight"
        self.cap, self.eps = conf.get("final_logit_softcapping"), conf.get("rms_norm_eps", 1e-6)
        try:  # sharded checkpoint
            wmap = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json")))["weight_map"]
        except Exception:  # single file
            wmap = {}

        def tensor(key):
            with safe_open(hf_hub_download(model_id, wmap.get(key, "model.safetensors")), "pt", device="cpu") as f:
                return f.get_tensor(key)

        self.W = tensor(unemb).to(device, torch.float32)          # (V, d)
        g = tensor("model.norm.weight").float()
        self.g = (1 + g if gemma else g).to(device)               # Gemma parameterizes g = 1 + w
        self.device = device

    def norm(self, h):
        h = h.to(self.device, torch.float32)
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

    def logits(self, h, chunk: int = 512):
        """Pre-cap logits (the cap is monotone, so the argmax and ranks are unchanged)."""
        return torch.cat([self.norm(h[i:i + chunk]) @ self.W.T for i in range(0, len(h), chunk)])

    def argmax(self, h, chunk: int = 512):
        return torch.cat([(self.norm(h[i:i + chunk]) @ self.W.T).argmax(-1) for i in range(0, len(h), chunk)])

    def logit_sd(self, h, chunk: int = 128):
        return torch.cat([(self.norm(h[i:i + chunk]) @ self.W.T).std(-1) for i in range(0, len(h), chunk)])

    def footprint(self, u):
        """u_V = W_U G u: how each token's logit moves per unit of gate."""
        return self.W @ (self.g * u.to(self.device, torch.float32))
