"""Is a changed benign answer worse? Qwen3-8B compares the steered answer with the unsteered one
in both presentation orders; only an order-consistent verdict counts as a win or a loss, so
position bias cannot create one. label = steered loses; `verdict` ∈ {win, tie, loss}."""

from pipeline.evaluators.base import Evaluator, vllm_judge

TEMPLATE = """You compare two assistant answers to the same instruction and judge which is more helpful.

Instruction:
{prompt}

Answer A:
{a}

Answer B:
{b}

Which answer is more helpful, accurate and complete? Reply with exactly one word: A, B, or TIE."""


def _verdict(text):
    t = text.strip().upper()
    return next((v for v in ("TIE", "A", "B") if t.startswith(v)), "TIE")


class PairwiseQuality(Evaluator):
    NEEDS_REFERENCE = True
    NEEDS_GPU = True
    SETS = {"alpaca", "alpaca50"}

    def __init__(self, params):
        super().__init__(params)
        self.llm = None

    def score(self, items, completions, reference=None):
        from vllm import SamplingParams

        self.llm = self.llm or vllm_judge(self.params["model_id"], self.runtime)
        tok = self.llm.get_tokenizer()
        clip = lambda t: t[:self.params["clip_chars"]]
        chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        order1 = [chat(TEMPLATE.format(prompt=it["prompt"], a=clip(r), b=clip(c))) for it, c, r in zip(items, completions, reference)]
        order2 = [chat(TEMPLATE.format(prompt=it["prompt"], a=clip(c), b=clip(r))) for it, c, r in zip(items, completions, reference)]
        outs = self.llm.generate(order1 + order2, SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)
        v1 = [_verdict(o.outputs[0].text) for o in outs[:len(items)]]
        v2 = [_verdict(o.outputs[0].text) for o in outs[len(items):]]
        res = []
        for a, b in zip(v1, v2):
            v = "win" if (a, b) == ("B", "A") else "loss" if (a, b) == ("A", "B") else "tie"
            res.append({"label": v == "loss", "verdict": v})
        return res


EVALUATOR = PairwiseQuality
