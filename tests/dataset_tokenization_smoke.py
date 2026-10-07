"""Check batched/cached tokenization against the original sample-wise logic."""

import itertools
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import get_dataset


LATENT_TOKENS = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
SPECS = [
    ("Qwen/Qwen3-1.7B-Base", False),
    ("ByteDance/Ouro-1.4B", True),
]


def main():
    raw = json.load(open("data/gsm_train.json"))[:512]
    for model_id, trust_remote_code in SPECS:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKENS})
        dataset = get_dataset(
            "data/gsm_train.json",
            tokenizer,
            max_size=512,
            cache_dir="data/tokenized_cache",
        )
        for index, sample in enumerate(raw):
            expected_question = tokenizer.encode(
                sample["question"] + "\n", add_special_tokens=True
            )
            expected_steps = [
                tokenizer.encode(step + "\n", add_special_tokens=False)
                for step in sample["steps"]
            ]
            expected_answer = tokenizer.encode(
                "### " + sample["answer"], add_special_tokens=False
            ) + [tokenizer.eos_token_id]
            actual = dataset[index]
            assert actual["question_tokenized"] == expected_question
            assert actual["steps_tokenized"] == expected_steps
            assert actual["answer_tokenized"] == expected_answer
            assert list(itertools.chain.from_iterable(expected_steps)) == list(
                itertools.chain.from_iterable(actual["steps_tokenized"])
            )
        print(f"Tokenization equivalence passed: {model_id} ({len(raw)} examples)")


if __name__ == "__main__":
    main()
