import os
import json
import ast
from typing import Dict, List
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL = "tiiuae/Falcon-H1-0.5B-Base"
DATASET = "recogna-nlp/UltrachatBR"
SPLIT = "train"  

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tokenizer.pad_token is None and tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token

def parse_conversa(raw: str) -> List[Dict[str, str]]:
    try:
        return json.loads(raw)
    except Exception:
        return ast.literal_eval(raw)

def build_text(turns: List[Dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = []
        for t in turns:
            if "humano" in t:
                messages.append({"role": "user", "content": t["humano"]})
            if "assistente" in t:
                messages.append({"role": "assistant", "content": t["assistente"]})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts = []
    for t in turns:
        if "humano" in t:
            parts.append(f"Usuário: {t['humano']}")
        if "assistente" in t:
            parts.append(f"Assistente: {t['assistente']}")
    eos = tokenizer.eos_token or ""
    return "\n".join(parts) + eos

def add_token_count(batch: dict[str, list[str]]) -> dict[str, list[int]]:
    enc = tokenizer(
        batch["conversa"],           # usa a string exatamente como vem
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_length=True,          # devolve apenas os comprimentos
    )
    return {"n_tokens": enc["length"]}

ds = load_dataset(DATASET, split=SPLIT)
ds = ds.map(add_token_count, batched=True, batch_size=512, num_proc=8)

total_tokens = int(np.sum(ds["n_tokens"]))
p50, p90, p95, p99 = np.percentile(ds["n_tokens"], [50, 90, 95, 99])

print(f"Total de tokens: {total_tokens:,}")
print(f"Mediana: {p50:.0f} | p90: {p90:.0f} | p95: {p95:.0f} | p99: {p99:.0f}")