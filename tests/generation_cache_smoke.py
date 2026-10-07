"""Real-model correctness and timing smoke test for Coconut generation caching."""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coconut import Coconut
from run import (
    LATENT_TOKENS,
    ensure_ouro_transformers_compatibility,
    initialize_latent_embeddings,
)


MODELS = {
    "qwen": {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "trust_remote_code": False,
    },
    "ouro": {
        "model_id": "ByteDance/Ouro-1.4B",
        "revision": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        "trust_remote_code": True,
    },
}


def load_coconut(name, attention_backend):
    spec = MODELS[name]
    config = AutoConfig.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
        trust_remote_code=spec["trust_remote_code"],
    )
    model_class = AutoModelForCausalLM
    if getattr(config, "model_type", None) == "ouro":
        model_class = ensure_ouro_transformers_compatibility(config)
        config.total_ut_steps = 1
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = config.eos_token_id
    model = model_class.from_pretrained(
        spec["model_id"],
        config=config,
        revision=spec["revision"],
        trust_remote_code=spec["trust_remote_code"],
        attn_implementation=attention_backend,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
        trust_remote_code=spec["trust_remote_code"],
    )
    original_vocab_size = len(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
    initialize_latent_embeddings(model, original_vocab_size)
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    model = Coconut(
        model,
        latent_id,
        start_id,
        end_id,
        tokenizer.eos_token_id,
    ).cuda().eval()
    return model, tokenizer


def timed_generate(model, input_ids, max_new_tokens, use_cache):
    torch.cuda.synchronize()
    start = time.monotonic()
    output = model.generate(
        input_ids,
        torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        use_generation_cache=use_cache,
    )
    torch.cuda.synchronize()
    return output, time.monotonic() - start


def run_model(name, attention_backend, max_new_tokens):
    model, tokenizer = load_coconut(name, attention_backend)
    question = tokenizer.encode(
        "Jan has 8 marbles and buys 5 more. How many marbles does Jan have?\n",
        add_special_tokens=True,
    )
    input_ids = torch.tensor(
        [
            question
            + [model.start_latent_id]
            + [model.latent_token_id] * 3
            + [model.end_latent_id]
        ],
        device="cuda",
    )

    # Warm kernels and allocator state before collecting either timing.
    with torch.inference_mode():
        model.generate(
            input_ids,
            torch.ones_like(input_ids),
            max_new_tokens=2,
            use_generation_cache=True,
        )
        uncached, uncached_seconds = timed_generate(
            model, input_ids, max_new_tokens, False
        )
        cached, cached_seconds = timed_generate(model, input_ids, max_new_tokens, True)

    if not torch.equal(cached, uncached):
        raise AssertionError(
            f"{name} cached tokens differ:\n"
            f"uncached={uncached.tolist()}\ncached={cached.tolist()}"
        )
    generated = cached.shape[1] - input_ids.shape[1]
    print(
        f"GENERATION_CACHE_PASS model={name} backend={attention_backend} "
        f"generated_tokens={generated} uncached_seconds={uncached_seconds:.4f} "
        f"cached_seconds={cached_seconds:.4f} "
        f"speedup={uncached_seconds / cached_seconds:.3f}x"
    )
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "ouro", "all"], default="all")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    names = MODELS if args.model == "all" else [args.model]
    for name in names:
        run_model(name, args.attention_backend, args.max_new_tokens)


if __name__ == "__main__":
    main()
