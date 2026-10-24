"""Which emotion does the story's main character feel? Qwen3-8B picks one of 12 emotions or
"neutral" (65.6% accurate on 160 labelled stories). label = the requested emotion is present
(for removal, report 1 − rate); `emotion` holds the judged label and `vocab` the rate of the
requested emotion's words per 100 words (a judge-free lexical proxy)."""

import re

from pipeline.evaluators.base import Evaluator, vllm_judge

LABELS = ["happy", "sad", "angry", "afraid", "guilty", "loving", "desperate", "inspired",
          "nervous", "proud", "calm", "surprised", "neutral"]
PROMPT = """Read the story below and decide which ONE emotion its main character most strongly feels.

Story:
{story}

Answer with exactly one word from this list: {labels}.
Use "neutral" if no emotion is clearly conveyed."""
LEXICON = {
    "happy": r"\b(happ\w*|joy\w*|delight\w*|cheer\w*|glee\w*|elat\w*|thrill\w*|grin\w*|beam\w*|laugh\w*|smil\w*)\b",
    "sad": r"\b(sad\w*|sorrow\w*|grie\w*|tear\w*|weep\w*|melanchol\w*|lonel\w*|mourn\w*|ache|ached|aching|heartbr\w*|despair\w*)\b",
    "angry": r"\b(anger|angr\w*|fur(y|ious\w*)|rage\w*|raging|seeth\w*|livid|fum\w*|resent\w*|clench\w*|snarl\w*|shout\w*|yell\w*)\b",
    "afraid": r"\b(afraid|fear\w*|terr\w*|panic\w*|dread\w*|scar\w*|fright\w*|trembl\w*|horr\w*|anxi\w*|alarm\w*)\b",
}


def _parse(text):
    t = text.strip().lower()
    return next((l for l in LABELS if t.startswith(l)), None) or next((l for l in LABELS if l in t[:40]), "neutral")


def vocab_rate(text: str, emotion: str) -> float:
    return 100 * len(re.findall(LEXICON[emotion], text.lower())) / max(1, len(text.split()))


class EmotionJudge(Evaluator):
    NEEDS_GPU = True
    SETS = {"ask_none", "ask_happy", "ask_sad", "ask_angry", "ask_afraid"}

    def __init__(self, params):
        super().__init__(params)
        self.llm = None

    def score(self, items, completions, reference=None):
        from vllm import SamplingParams

        self.llm = self.llm or vllm_judge(self.params["model_id"], self.runtime)
        tok = self.llm.get_tokenizer()
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(story=c[:self.params["clip_chars"]], labels=", ".join(LABELS))}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in completions]
        outs = self.llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6), use_tqdm=False)
        res = []
        for it, c, o in zip(items, completions, outs):
            lab, asked = _parse(o.outputs[0].text), it.get("asked")
            res.append({"label": lab == asked, "emotion": lab, **({"vocab": vocab_rate(c, asked)} if asked else {})})
        return res


EVALUATOR = EmotionJudge
