import re
from vocab import load_vocab, vocab_fraction

VOCAB = load_vocab()


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def correctness_score(completion: str, aliases: list[str]) -> float:
    # 0.0 = źle, 0.5 = częściowe dopasowanie aliasu, 1.0 = dokładnie
    norm_c = _normalise(completion)
    score = 0.0
    for alias in aliases:
        norm_a = _normalise(alias)
        if not norm_a:
            continue
        if norm_a == norm_c:
            return 1.0
        if re.search(rf"\b{re.escape(norm_a)}\b", norm_c):
            score = max(score, 0.5)
    return score


def reward_vocab(
    completions: list[str],
    **kwargs,
) -> list[float]:
    return [vocab_fraction(c, VOCAB) * 0.4 for c in completions]


def reward_correctness(
    completions: list[str],
    answer_aliases: list[list[str]],
    **kwargs,
) -> list[float]:
    return [
        correctness_score(c, aliases) * 1.6
        for c, aliases in zip(completions, answer_aliases)
    ]


def reward_anti_hack(
    completions: list[str],
    answer_aliases: list[list[str]],
    **kwargs,
) -> list[float]:
    scores = []
    for completion, aliases in zip(completions, answer_aliases):
        penalty = 0.0
        corr = correctness_score(completion, aliases)
        vocab = vocab_fraction(completion, VOCAB)
        tokens = _tokenise(completion)

        if corr == 0.0 and vocab > 0.6:
            penalty += 0.3 * vocab

        if corr == 0.0 and len(tokens) > 8:
            penalty += 0.08 * (len(tokens) - 8)

        if tokens:
            max_repeat = max(tokens.count(t) for t in set(tokens))
            if max_repeat >= 4:
                penalty += 0.15 * (max_repeat - 3)

        scores.append(-penalty)
    return scores
