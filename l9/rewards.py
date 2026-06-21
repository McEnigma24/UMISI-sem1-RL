import re
from vocab import load_vocab, vocab_fraction

VOCAB = load_vocab()


def reward_vocab(
    completions: list[str],
    **kwargs,
) -> list[float]:
    return [vocab_fraction(c, VOCAB) * 0.4 for c in completions]


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def reward_correctness(
    completions: list[str],
    answer_aliases: list[list[str]],
    **kwargs,
) -> list[float]:
    scores = []
    for completion, aliases in zip(completions, answer_aliases):
        norm_c = _normalise(completion)
        score = 0.0
        for alias in aliases:
            norm_a = _normalise(alias)
            if not norm_a:
                continue
            if norm_a == norm_c:
                score = 1.0
                break
            if norm_a in norm_c:
                score = max(score, 0.5)
        scores.append(score * 1.6)
    return scores
