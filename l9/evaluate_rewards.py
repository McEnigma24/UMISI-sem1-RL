from rewards import (
    correctness_score,
    reward_anti_hack,
    reward_correctness,
    reward_vocab,
)

ALIASES = [["paris"], ["102"], ["jupiter"], ["366"], ["carbon dioxide", "co2"]]

HACKING = [
    "the the the the the the the the the the the the",
    "and and and and and and and and and and and and and and and",
    "this is a very simple answer with many common words but no fact",
    "paris paris paris paris paris paris paris paris paris paris",
]

HONEST = [
    "paris",
    "102",
    "jupiter",
    "366",
    "co2",
]


def total_reward(completions, aliases, use_anti_hack: bool):
    v = reward_vocab(completions)
    c = reward_correctness(completions, aliases)
    if use_anti_hack:
        a = reward_anti_hack(completions, aliases)
        return [vi + ci + ai for vi, ci, ai in zip(v, c, a)]
    return [vi + ci for vi, ci in zip(v, c)]


def main():
    print("=== Synthetic hacking examples ===")
    for text in HACKING:
        corr = correctness_score(text, ALIASES[0])
        print(f"\n{text[:60]}...")
        print(f"  correctness={corr:.2f}  vocab={reward_vocab([text])[0]:.3f}")
        print(f"  baseline total={total_reward([text], [ALIASES[0]], False)[0]:.3f}")
        print(f"  with anti_hack={total_reward([text], [ALIASES[0]], True)[0]:.3f}")
        print(f"  anti_hack only={reward_anti_hack([text], [ALIASES[0]])[0]:.3f}")

    print("\n=== Honest short answers ===")
    for text, aliases in zip(HONEST, ALIASES):
        print(
            f"{text:6} baseline={total_reward([text], [aliases], False)[0]:.3f}  "
            f"anti_hack={total_reward([text], [aliases], True)[0]:.3f}"
        )


if __name__ == "__main__":
    main()
