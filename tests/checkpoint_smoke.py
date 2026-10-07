"""Small multi-GPU round-trip test for the FSDP training checkpoint format."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import save_fsdp_checkpoint


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 4),
    ).to(local_rank)
    model = FSDP(model, device_id=local_rank, use_orig_params=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    inputs = torch.arange(32, device=local_rank, dtype=torch.float32).view(4, 8)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()

    checkpoint_path = os.environ.get(
        "CHECKPOINT_SMOKE_PATH", "/tmp/coconut-checkpoint-smoke.pt"
    )
    save_fsdp_checkpoint(
        model,
        optimizer,
        checkpoint_path,
        rank,
        world_size,
        completed_epochs=3,
        total_train_steps=17,
        best_acc=0.25,
        save_optimizer_state=True,
    )
    dist.barrier()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["format_version"] == 2
    assert checkpoint["completed_epochs"] == 3
    assert checkpoint["total_train_steps"] == 17
    assert len(checkpoint["rng_states"]) == world_size

    torch.manual_seed(0)
    restored = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 4),
    )
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored = FSDP(restored.to(local_rank), device_id=local_rank, use_orig_params=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)

    optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(
        restored,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_config,
    ):
        optimizer_state = FSDP.optim_state_dict_to_load(
            restored, restored_optimizer, checkpoint["optimizer_state_dict"]
        )
    restored_optimizer.load_state_dict(optimizer_state)
    assert restored_optimizer.state

    with torch.no_grad():
        actual = restored(inputs)
    expected = model(inputs).detach()
    torch.testing.assert_close(actual, expected)

    dist.barrier()
    if rank == 0:
        os.remove(checkpoint_path)
        print("FSDP checkpoint save/load round trip passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
