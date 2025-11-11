from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser
from dataclasses import dataclass
import ast
import json
import re


@dataclass
class DataSubsetArgs:
    dataset_sample_size: int | None = None
    dataset_eval_fraction: float | None = None


_KEY_RE = re.compile(
    r"'?\s*(humano|Humano|human|Human|assistente|Assistente|assistant|Assistant)\s*'?\s*:\s*",
    re.S,
)

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s

def _to_messages(ex: dict) -> dict:
    s = ex["conversa"]
    msgs: list[dict[str, str]] = []
    hits = list(_KEY_RE.finditer(s))
    if not hits:
        return {"messages": [{"role": "assistant", "content": s}]}
    for i, m in enumerate(hits):
        role_raw = m.group(1).lower()
        role = "user" if role_raw in {"humano", "human"} else "assistant"
        start = m.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(s)
        chunk = s[start:end].strip().rstrip(",}] ")
        msgs.append({"role": role, "content": _strip_quotes(chunk)})
    return {"messages": msgs}

def main(script_args, training_args, model_args, subset_args):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% if bos_token %}{{ bos_token }}{% endif %}"
            "{% for m in messages %}"
            "{% if m['role'] == 'user' %}[user]\n{{ m['content'] }}\n"
            "{% elif m['role'] == 'assistant' %}[assistant]\n{% generation %}{{ m['content'] }}{% endgeneration %}\n"
            "{% elif m['role'] == 'system' %}[system]\n{{ m['content'] }}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}[assistant]\n{% endif %}"
        )

    seed = getattr(training_args, "seed", 42)
    sample_size = getattr(subset_args, "dataset_sample_size", 200_000)
    eval_fraction = getattr(subset_args, "dataset_eval_fraction", 0.05)
    num_proc = getattr(script_args, "dataset_num_proc", 16)

    base = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split="train",
        num_proc=num_proc,
    )
    base = base.shuffle(seed=seed)
    sample_size = min(sample_size, len(base))
    base = base.select(range(sample_size))

    splits = base.train_test_split(test_size=eval_fraction, seed=seed, shuffle=False)
    dataset_train = splits["train"]
    dataset_eval = splits["test"] if training_args.eval_strategy != "no" else None
    
    dataset_train = dataset_train.map(_to_messages, num_proc=num_proc)
    if dataset_eval is not None:
        dataset_eval = dataset_eval.map(_to_messages, num_proc=num_proc)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, DataSubsetArgs))
    script_args, training_args, model_args, subset_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args, subset_args)