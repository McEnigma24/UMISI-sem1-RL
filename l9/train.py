import argparse
import json
import os

from trl import GRPOTrainer, GRPOConfig

import torch
from peft import LoraConfig
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from data import build_dataset
from rewards import reward_correctness, reward_vocab
from vocab import load_vocab, vocab_fraction

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
RUNS_BASE = "runs"


def next_run_dir(suffix: str) -> str:
    os.makedirs(RUNS_BASE, exist_ok=True)
    existing_numbers = [
        int(d[:6])
        for d in os.listdir(RUNS_BASE)
        if os.path.isdir(os.path.join(RUNS_BASE, d))
        and len(d) >= 6
        and d[:6].isdigit()
    ]
    next_n = max(existing_numbers, default=0) + 1
    return os.path.join(RUNS_BASE, f"{next_n:06d}_{suffix}")


def setup_run(suffix: str, config: dict) -> str:
    run_dir = next_run_dir(suffix)
    for subdir in ("conversations", "tensorboard", "checkpoints"):
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nRun directory : {run_dir}")
    print(f"Config saved  : {config_path}\n")
    return run_dir


class ConversationLoggerCallback(TrainerCallback):
    PROBE_QUESTIONS = [
        "What is the capital of France?",
        "How does the sun make light?",
        "Why is the sky blue?",
        "What is 17 multiplied by 6?",
        "Who wrote Romeo and Juliet?",
    ]

    def __init__(
        self,
        tokenizer,
        vocab: set[str],
        run_dir: str,
        log_every_steps: int = 50,
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.conv_dir = os.path.join(run_dir, "conversations")
        self.tb_dir = os.path.join(run_dir, "tensorboard")
        self.log_every_steps = log_every_steps
        self.tb_writer: SummaryWriter | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every_steps != 0:
            return
        if model is None:
            return

        was_training = model.training
        model.eval()

        scores = []
        records = []

        for question in self.PROBE_QUESTIONS:
            msgs = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            attention_mask = (input_ids != self.tokenizer.eos_token_id).long()

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=80,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            score = vocab_fraction(response, self.vocab)
            scores.append(score)
            records.append({
                "step": state.global_step,
                "question": question,
                "response": response,
                "vocab_score": round(score, 3),
            })

        if was_training:
            model.train()

        path = os.path.join(self.conv_dir, f"step_{state.global_step:05d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        if self.tb_writer:
            mean = sum(scores) / len(scores)
            self.tb_writer.add_scalar("vocab/probe_mean", mean, state.global_step)
            for i, s in enumerate(scores):
                self.tb_writer.add_scalar(f"vocab/probe_{i}", s, state.global_step)
            self.tb_writer.flush()

        print(f"\n[step {state.global_step}] vocab probe mean: {sum(scores)/len(scores):.3f}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()


def parse_args():
    p = argparse.ArgumentParser(description="GRPO+LoRA fine-tuning — SmolLM2-135M")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="20-step smoke test with reduced settings",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training even when a GPU is available",
    )
    p.add_argument(
        "--name",
        default=None,
        metavar="SUFFIX",
        help="Human-readable suffix for the run directory (auto-generated if omitted)",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume training from a saved checkpoint",
    )
    p.add_argument(
        "--n-train",
        type=int,
        default=3000,
        help="Number of training examples to use (default: 3000)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log probe conversations every N steps (default: 50)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_gpu = torch.cuda.is_available() and not args.cpu
    dtype = torch.bfloat16 if use_gpu else torch.float32
    device_map = "auto" if use_gpu else "cpu"

    lora_params = dict(r=8, lora_alpha=16, lora_dropout=0.05)

    if args.smoke:
        grpo_params = dict(
            max_steps=20,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_completion_length=40,
            learning_rate=5e-5,
            beta=0.05,
            bf16=use_gpu,
            gradient_checkpointing=True,
            log_completions=True,
            num_completions_to_print=0,
            logging_steps=5,
            save_steps=20,
            report_to="tensorboard",
            dataloader_num_workers=0,
        )
        log_every = 5
        n_train = 20
    else:
        grpo_params = dict(
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_generations=8,
            max_completion_length=80,
            learning_rate=5e-5,
            beta=0.05,
            bf16=use_gpu,
            gradient_checkpointing=True,
            log_completions=True,
            num_completions_to_print=0,
            logging_steps=10,
            save_steps=100,
            report_to="tensorboard",
            dataloader_num_workers=0,
        )
        log_every = args.log_every
        n_train = args.n_train

    if args.name:
        suffix = args.name
    elif args.smoke:
        suffix = "smoke"
    else:
        lr = grpo_params["learning_rate"]
        suffix = (
            f"full"
            f"_g{grpo_params['num_generations']}"
            f"_r{lora_params['r']}"
            f"_lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
        )

    config = {
        "model": MODEL_ID,
        "mode": "smoke" if args.smoke else "full",
        "device": "cpu" if not use_gpu else device_map,
        "dtype": str(dtype).replace("torch.", ""),
        "resumed_from": args.resume,
        "n_train": n_train,
        "lora": lora_params,
        "grpo": {k: v for k, v in grpo_params.items()
                 if k not in ("report_to", "dataloader_num_workers",
                              "log_completions", "num_completions_to_print")},
    }
    run_dir = setup_run(suffix, config)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")

    print(f"Loading model ({dtype}, {device_map}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device_map,
    )
    print(f"  dtype={next(model.parameters()).dtype}  device={next(model.parameters()).device}\n")

    dataset = build_dataset(tokenizer, n_train=n_train)

    peft_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=lora_params["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_args = GRPOConfig(output_dir=checkpoint_dir, **grpo_params)

    vocab = load_vocab()
    callback = ConversationLoggerCallback(
        tokenizer=tokenizer,
        vocab=vocab,
        run_dir=run_dir,
        log_every_steps=log_every,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_vocab, reward_correctness],
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[callback],
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(os.path.join(checkpoint_dir, "final"))
    print(f"\nTraining complete. Final adapter: {checkpoint_dir}/final")


if __name__ == "__main__":
    main()
