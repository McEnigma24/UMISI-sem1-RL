from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


def build_dataset(
    tokenizer: PreTrainedTokenizerBase,
    n_train: int = 3000,
    max_prompt_tokens: int = 256,
    max_answer_words: int = 5,
) -> Dataset:
    print("Downloading TriviaQA (rc.nocontext) ...")
    raw = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")

    def process(example):
        if len(example["answer"]["value"].split()) > max_answer_words:
            return {"keep": False, "prompt": "", "answer_aliases": []}

        msgs = [{"role": "user", "content": example["question"]}]
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )

        if len(tokenizer.encode(prompt)) > max_prompt_tokens:
            return {"keep": False, "prompt": "", "answer_aliases": []}

        return {
            "keep": True,
            "prompt": prompt,
            "answer_aliases": example["answer"]["aliases"],
        }

    ds = raw.map(
        process,
        remove_columns=raw.column_names,
        desc="Formatting prompts",
    )
    ds = ds.filter(lambda x: x["keep"], desc="Filtering")
    ds = ds.remove_columns(["keep"])
    ds = ds.select(range(min(n_train, len(ds))))

    print(f"Dataset ready: {len(ds)} examples")
    return ds
