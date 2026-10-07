# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.distributed
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from coconut import Coconut
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
import importlib
import random
import re
import time
import numpy as np
from utils import Config, set_seed


LATENT_TOKENS = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]


def config_value(configs, name, default=None):
    return getattr(configs, name, default)


def has_checkpoint(path):
    return path not in (None, "None", "null", "")


def get_scheduled_stage(epoch, configs):
    """Map a global epoch to a curriculum stage.

    ``stage_epochs`` allows a longer stage 0 followed by shorter transition
    stages, e.g. [3, 1, 1, 1, 1] for the larger-model GSM8K recipe. Once the
    listed schedule is exhausted, training remains in the final listed stage.
    """
    if configs.cot or configs.no_cot:
        return 0
    stage_epochs = config_value(configs, "stage_epochs")
    if stage_epochs:
        elapsed = 0
        for stage, epochs_in_stage in enumerate(stage_epochs):
            elapsed += epochs_in_stage
            if epoch < elapsed:
                return stage
        return len(stage_epochs) - 1
    return epoch // configs.epochs_per_stage


def get_decoder_layer_classes(model):
    base_model = model.base_causallm if isinstance(model, Coconut) else model
    decoder = getattr(base_model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is not None and len(layers) > 0:
        return {layers[0].__class__}
    transformer = getattr(base_model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if blocks is not None and len(blocks) > 0:
        return {blocks[0].__class__}
    raise ValueError(f"Cannot locate decoder layers in {type(base_model).__name__}")


def causal_lm_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )


def create_optimizer(parameters, configs):
    kwargs = {
        "lr": configs.lr,
        "weight_decay": configs.weight_decay,
    }
    if config_value(configs, "fused_optimizer", False):
        kwargs["fused"] = True
    return optim.AdamW(parameters, **kwargs)


def forward_for_loss(parallel_model, batch, is_ouro, ouro_recurrent_steps):
    """Use final recurrent-step logits for direct Ouro baselines.

    Ouro's stock labeled forward uses a learned weighted mixture of all exits,
    while Coconut uses final-step logits. Computing the loss here keeps the
    CoT and Coconut readout objectives aligned.
    """
    if not is_ouro:
        return parallel_model(**batch).loss
    labels = batch["labels"]
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = parallel_model(
        **model_inputs,
        use_cache=False,
        exit_at_step=ouro_recurrent_steps - 1,
    )
    return causal_lm_loss(outputs.logits, labels)


def initialize_latent_embeddings(model, original_vocab_size):
    input_embeddings = model.get_input_embeddings().weight
    output_layer = model.get_output_embeddings()
    with torch.no_grad():
        input_mean = input_embeddings[:original_vocab_size].mean(dim=0)
        input_embeddings[original_vocab_size:].copy_(input_mean)
        if (
            output_layer is not None
            and output_layer.weight.data_ptr() != input_embeddings.data_ptr()
        ):
            output_mean = output_layer.weight[:original_vocab_size].mean(dim=0)
            output_layer.weight[original_vocab_size:].copy_(output_mean)


def ensure_ouro_transformers_compatibility(model_config):
    """Bridge Ouro's pinned remote code to Transformers 5.x RoPE APIs."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = (
            LlamaRotaryEmbedding.compute_default_rope_parameters
        )

    ouro_module_name = model_config.__class__.__module__.replace(
        "configuration_ouro", "modeling_ouro"
    )
    ouro_module = importlib.import_module(ouro_module_name)
    ouro_rotary_class = ouro_module.OuroRotaryEmbedding
    if not hasattr(ouro_rotary_class, "compute_default_rope_parameters"):
        ouro_rotary_class.compute_default_rope_parameters = staticmethod(
            LlamaRotaryEmbedding.compute_default_rope_parameters
        )

    def adapt_mask_function(mask_function):
        @functools.wraps(mask_function)
        def compatible_mask_function(*args, **kwargs):
            if "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)
            return mask_function(*args, **kwargs)

        return compatible_mask_function

    ouro_module.create_causal_mask = adapt_mask_function(
        ouro_module.create_causal_mask
    )
    ouro_module.create_sliding_window_causal_mask = adapt_mask_function(
        ouro_module.create_sliding_window_causal_mask
    )
    return ouro_module.OuroForCausalLM


def checkpoint_epoch(filename):
    match = re.fullmatch(r"checkpoint_(\d+)", filename)
    return int(match.group(1)) if match else None


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(states, rank, world_size):
    if not states:
        return
    if len(states) != world_size:
        if rank == 0:
            print(
                "Warning: checkpoint world size differs from this run; "
                "RNG state will not be restored."
            )
        return
    state = states[rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    parallel_model,
    optimizer,
    checkpoint_path,
    rank,
    world_size,
    completed_epochs,
    total_train_steps,
    best_acc,
    save_optimizer_state,
):
    if isinstance(parallel_model, FSDP):
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_state_config = FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            parallel_model,
            StateDictType.FULL_STATE_DICT,
            state_config,
            optim_state_config,
        ):
            model_state = parallel_model.state_dict()
            optimizer_state = (
                FSDP.optim_state_dict(parallel_model, optimizer)
                if save_optimizer_state
                else None
            )
    elif rank == 0:
        model_state = {
            key: value.detach().cpu()
            for key, value in parallel_model.module.state_dict().items()
        }
        optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    else:
        model_state = None
        optimizer_state = None

    rng_states = [None] * world_size
    dist.all_gather_object(rng_states, capture_rng_state())

    if rank == 0:
        checkpoint = {
            "format_version": 2,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "completed_epochs": completed_epochs,
            "total_train_steps": total_train_steps,
            "best_acc": best_acc,
            "rng_states": rng_states,
            "world_size": world_size,
        }
        temporary_path = f"{checkpoint_path}.tmp"
        with open(temporary_path, "wb") as checkpoint_file:
            torch.save(checkpoint, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, checkpoint_path)
        print(f"Saved training state to {checkpoint_path}")
    del model_state, optimizer_state, rng_states


# Keep the old public name for the standalone checkpoint round-trip test and
# any existing callers outside this repository.
save_fsdp_checkpoint = save_training_checkpoint


def get_wandb_run_id(save_dir):
    run_id_path = os.path.join(save_dir, "wandb_run_id")
    if os.path.exists(run_id_path):
        with open(run_id_path) as run_id_file:
            return run_id_file.read().strip()

    run_id = wandb.util.generate_id()
    temporary_path = f"{run_id_path}.tmp"
    with open(temporary_path, "w") as run_id_file:
        run_id_file.write(run_id)
        run_id_file.flush()
        os.fsync(run_id_file.fileno())
    os.replace(temporary_path, run_id_path)
    return run_id


def main():

    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # init distributed environment
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank)
    )

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)
    checkpoint_epochs = sorted(
        (checkpoint_epoch(filename), filename)
        for filename in cur_ckpts
        if checkpoint_epoch(filename) is not None
    )
    resuming_existing_run = False

    # check if the job is preempted and resumed.

    if checkpoint_epochs and not configs.only_eval:
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        latest_epoch, latest_checkpoint = checkpoint_epochs[-1]
        configs.resume = latest_epoch
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        resuming_existing_run = True
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        if not has_checkpoint(configs.load_model_path):
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # not an intended use case at this point
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    trust_remote_code = config_value(configs, "trust_remote_code", False)
    model_revision = config_value(configs, "model_revision", "main")
    model_config = AutoConfig.from_pretrained(
        configs.model_id,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
    )
    ouro_model_class = None
    if getattr(model_config, "model_type", None) == "ouro":
        ouro_model_class = ensure_ouro_transformers_compatibility(model_config)
        if not hasattr(model_config, "pad_token_id"):
            model_config.pad_token_id = model_config.eos_token_id
        model_config.total_ut_steps = config_value(
            configs, "ouro_recurrent_steps", model_config.total_ut_steps
        )
        if model_config.total_ut_steps < 1:
            raise ValueError("ouro_recurrent_steps must be at least 1")
    load_kwargs = {
        "config": model_config,
        "revision": model_revision,
        "trust_remote_code": trust_remote_code,
    }
    attn_implementation = config_value(configs, "attn_implementation")
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model_loader = ouro_model_class or AutoModelForCausalLM
    model = model_loader.from_pretrained(configs.model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_id,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
    )
    original_vocab_size = len(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKENS})
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    is_ouro = getattr(model.config, "model_type", None) == "ouro"
    ouro_recurrent_steps = getattr(model.config, "total_ut_steps", 1)

    if rank == 0:
        print(
            f"Loaded {configs.model_id}@{model_revision} "
            f"(model_type={model.config.model_type}, "
            f"ouro_recurrent_steps={ouro_recurrent_steps if is_ouro else 'n/a'})"
        )

    loaded = False
    resume_state = None

    if has_checkpoint(configs.load_model_path):
        saved_checkpoint = torch.load(
            configs.load_model_path, map_location="cpu", weights_only=False
        )
        if (
            isinstance(saved_checkpoint, dict)
            and saved_checkpoint.get("format_version") == 2
            and "model_state_dict" in saved_checkpoint
        ):
            saved_weights = saved_checkpoint.pop("model_state_dict")
            if resuming_existing_run:
                resume_state = saved_checkpoint
        else:
            # Backward compatibility with the original weight-only checkpoints.
            saved_weights = saved_checkpoint

        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # we are loading a base model into coconut model
            # e.g., for GSM8k, we used a SFTed model to skip the stage 0
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            # loading from preempted run
            # will handle later
            pass

        else:
            # resume or evaluate sft model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # if we need new tokens, initialize their embeddings and lm heads
        model.resize_token_embeddings(len(tokenizer))
        initialize_latent_embeddings(model, original_vocab_size)

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if has_checkpoint(configs.load_model_path) and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    if has_checkpoint(configs.load_model_path):
        del saved_weights
        if not resuming_existing_run:
            del saved_checkpoint

    distributed_strategy = (
        "ddp"
        if configs.only_eval
        else config_value(configs, "distributed_strategy", "fsdp").lower()
    )
    if distributed_strategy not in {"fsdp", "ddp"}:
        raise ValueError("distributed_strategy must be either 'fsdp' or 'ddp'")
    print(
        f"Running {distributed_strategy.upper()} on rank = {rank}, "
        f"world size = {world_size}"
    )
    base_model = model.base_causallm if isinstance(model, Coconut) else model
    if config_value(configs, "gradient_checkpointing", False):
        try:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

    decoder_layer_classes = get_decoder_layer_classes(model)
    decoder_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=decoder_layer_classes,
    )
    model = model.to(local_rank)
    mixed_precision = None
    if configs.bf16:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    if distributed_strategy == "fsdp":
        parallel_model = FSDP(
            model,
            auto_wrap_policy=decoder_auto_wrap_policy,
            device_id=local_rank,
            mixed_precision=mixed_precision,
            use_orig_params=True,
        )
    else:
        parallel_model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=is_ouro,
        )

    del model

    if rank == 0:
        print(parallel_model)

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    max_val_samples = config_value(
        configs, "max_val_samples", 32 if configs.debug else 100000000
    )
    base_dataset_valid = get_dataset(
        configs.val_path,
        tokenizer,
        max_size=max_val_samples,
        cache_dir=config_value(configs, "tokenized_cache_dir", "data/tokenized_cache"),
    )

    if not configs.only_eval:
        max_train_samples = config_value(
            configs, "max_train_samples", 5000 if configs.debug else 100000000
        )
        base_dataset_train = get_dataset(
            configs.train_path,
            tokenizer,
            max_size=max_train_samples,
            cache_dir=config_value(
                configs, "tokenized_cache_dir", "data/tokenized_cache"
            ),
        )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = (
        resume_state.get("total_train_steps", 0) if resume_state else 0
    )

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(
            project=configs.project,
            name=configs.name,
            id=get_wandb_run_id(save_dir),
            resume="allow",
        )
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None

    else:
        optimizer = create_optimizer(parallel_model.parameters(), configs)

    best_acc = resume_state.get("best_acc", 0) if resume_state else 0

    if resume_state and resume_state.get("optimizer_state_dict") is not None:
        if configs.reset_optimizer:
            if rank == 0:
                print("Not restoring optimizer because reset_optimizer is enabled.")
        else:
            if distributed_strategy == "fsdp":
                optim_state_config = FullOptimStateDictConfig(
                    offload_to_cpu=True, rank0_only=False
                )
                with FSDP.state_dict_type(
                    parallel_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                    optim_state_config,
                ):
                    optimizer_state = FSDP.optim_state_dict_to_load(
                        parallel_model,
                        optimizer,
                        resume_state["optimizer_state_dict"],
                    )
            else:
                optimizer_state = resume_state["optimizer_state_dict"]
            optimizer.load_state_dict(optimizer_state)
            del optimizer_state
            if rank == 0:
                print("Restored optimizer state.")

    if resume_state:
        restore_rng_state(resume_state.get("rng_states"), rank, world_size)
        del resume_state

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    for epoch in range(configs.resume, configs.num_epochs):

        scheduled_stage = get_scheduled_stage(epoch, configs)
        if rank == 0:
            print(f"Epoch {epoch + 1}: curriculum stage {scheduled_stage}")
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            cache_dir=config_value(configs, "stage_cache_dir", "data/stage_cache"),
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:

            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
                cache_dir=config_value(configs, "stage_cache_dir", "data/stage_cache"),
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )
            train_dataloader.sampler.set_epoch(epoch)

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                cache_dir=config_value(configs, "stage_cache_dir", "data/stage_cache"),
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer:
                del optimizer

                optimizer = create_optimizer(parallel_model.parameters(), configs)

            parallel_model.module.train()

            torch.cuda.synchronize(local_rank)
            torch.cuda.reset_peak_memory_stats(local_rank)
            train_start_time = time.monotonic()
            local_examples_seen = 0
            profiler = None
            if rank == 0 and config_value(configs, "profile_training", False):
                profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                )
                profiler.start()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):

                local_examples_seen += batch["input_ids"].shape[0]

                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # copy the table due to a bug in wandb
                    # https://github.com/wandb/wandb/issues/2981

                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(local_rank)
                    for key in batch.keys()
                    if key != "idx"
                }

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=configs.bf16 and distributed_strategy == "ddp",
                ):
                    if configs.coconut:
                        batch_loss = parallel_model(
                            **batch, output_full_logits=False
                        ).loss
                    else:
                        batch_loss = forward_for_loss(
                            parallel_model, batch, is_ouro, ouro_recurrent_steps
                        )
                loss = batch_loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
                if profiler is not None:
                    profiler.step()
            pbar.close()
            if profiler is not None:
                profiler.stop()
                print(
                    profiler.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=30
                    )
                )
            torch.cuda.synchronize(local_rank)
            train_elapsed = torch.tensor(
                time.monotonic() - train_start_time,
                dtype=torch.float64,
                device=local_rank,
            )
            examples_seen = torch.tensor(
                local_examples_seen, dtype=torch.int64, device=local_rank
            )
            dist.all_reduce(train_elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(examples_seen, op=dist.ReduceOp.SUM)
            peak_memory = torch.tensor(
                torch.cuda.max_memory_allocated(local_rank),
                dtype=torch.int64,
                device=local_rank,
            )
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(
                    "Training throughput: "
                    f"{examples_seen.item() / train_elapsed.item():.2f} examples/s "
                    f"({examples_seen.item()} examples in {train_elapsed.item():.2f}s); "
                    f"peak allocated memory: {peak_memory.item() / 2**30:.2f} GiB/GPU"
                )
            dist.barrier()

            if not configs.save_only_improve and not configs.debug:
                save_training_checkpoint(
                    parallel_model,
                    optimizer,
                    os.path.join(save_dir, f"checkpoint_{epoch + 1}"),
                    rank,
                    world_size,
                    epoch + 1,
                    total_train_steps,
                    best_acc,
                    not configs.reset_optimizer,
                )
                dist.barrier()
                gc.collect()
                torch.cuda.empty_cache()

            if config_value(configs, "skip_validation", False):
                continue

            # val loss
            total_loss = 0

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):

                    batch = {
                        key: batch[key].to(local_rank)
                        for key in batch.keys()
                        if key != "idx"
                    }

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=configs.bf16 and distributed_strategy == "ddp",
                    ):
                        if configs.coconut:
                            loss = parallel_model(
                                **batch, output_full_logits=False
                            ).loss
                        else:
                            loss = forward_for_loss(
                                parallel_model, batch, is_ouro, ouro_recurrent_steps
                            )
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                if wandb_run and rank == 0:

                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        # val generation accuracy
        total_length = len(valid_gen_dataloader)

        pbar = tqdm(
            colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        )
        cor, cor_cot, total = (
            torch.tensor(0, device=local_rank),
            torch.tensor(0, device=local_rank),
            torch.tensor(0, device=local_rank),
        )

        with torch.no_grad():
            parallel_model.module.eval()
            torch.cuda.synchronize(local_rank)
            generation_start_time = time.monotonic()
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(local_rank)
                    for k, v in batch.items()
                    if v != None and k not in ["idx", "position_ids"]
                }
                # https://github.com/huggingface/transformers/issues/32492

                assert len(batch["input_ids"]) == 1
                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question = question_val[test_idx.cpu().item()]

                total += 1

                # synced_gpus=True in FSDP mode, as we need to keep # forward pass the same on each device
                generation_kwargs = {
                    **batch,
                    "max_new_tokens": max_new_tokens,
                    "synced_gpus": distributed_strategy == "fsdp",
                }
                if configs.coconut:
                    generation_kwargs["use_generation_cache"] = config_value(
                        configs, "validation_kv_cache", False
                    )
                if distributed_strategy == "ddp":
                    outputs = parallel_model.module.generate(**generation_kwargs)
                else:
                    # ``generate`` is a method on the wrapped Hugging Face model
                    # and bypasses the root FSDP forward hook. Materialize only
                    # the root-owned parameters (embeddings and LM head); child
                    # decoder-layer FSDP wrappers unshard themselves normally.
                    with FSDP.summon_full_params(
                        parallel_model, recurse=False, writeback=False
                    ):
                        outputs = parallel_model.module.generate(**generation_kwargs)

                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = (
                    ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                )

                if idx < 5 and rank == 0:
                    # print some examples
                    print(
                        f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                    )
                    print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")

                cor += answer_output == answer
                cor_cot += cot_output == answer_cot

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                )

            pbar.close()
            torch.cuda.synchronize(local_rank)
            generation_elapsed = torch.tensor(
                time.monotonic() - generation_start_time,
                dtype=torch.float64,
                device=local_rank,
            )
            dist.all_reduce(generation_elapsed, op=dist.ReduceOp.MAX)
            print(f"Device {rank}: Cor={cor}, CoT={cor_cot}, Total={total}")

        dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
        dist.all_reduce(cor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        cor_cot = cor_cot.item()
        cor = cor.item()
        total = total.item()
        if rank == 0:
            print(
                "Generation throughput: "
                f"{total / generation_elapsed.item():.2f} examples/s "
                f"({total} examples in {generation_elapsed.item():.2f}s)"
            )
            print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
            print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")
        sys.stdout.flush()

        if wandb_run:
            wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})

        if configs.only_eval:
            break

        dist.barrier()
        improved = cor / total > best_acc
        if improved:
            best_acc = cor / total
        should_save = not configs.debug and configs.save_only_improve and improved
        if should_save:
            save_training_checkpoint(
                parallel_model,
                optimizer,
                os.path.join(save_dir, f"checkpoint_{epoch + 1}"),
                rank,
                world_size,
                epoch + 1,
                total_train_steps,
                best_acc,
                not configs.reset_optimizer,
            )

            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
