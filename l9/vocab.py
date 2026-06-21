import re
from wordfreq import top_n_list

_vocab_cache: set[str] | None = None


def load_vocab(n: int = 1000) -> set[str]:
    global _vocab_cache
    if _vocab_cache is None:
        _vocab_cache = set(top_n_list("en", n))
    return _vocab_cache


def vocab_fraction(text: str, vocab: set[str]) -> float:
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in vocab) / len(tokens)
