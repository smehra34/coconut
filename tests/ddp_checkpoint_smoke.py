"""Small multi-GPU round-trip test for DDP training checkpoints."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import save_training_checkpoint


def build_model(device):
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 4),
    ).to(device)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    torch.manual_seed(1234)
    model = DDP(build_model(local_rank), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.arange(32, device=local_rank, dtype=torch.float32).view(4, 8)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()

    checkpoint_path = os.environ.get(
        "CHECKPOINT_SMOKE_PATH", "/tmp/coconut-ddp-checkpoint-smoke.pt"
    )
    save_training_checkpoint(
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
    torch.manual_seed(0)
    restored = build_model(local_rank)
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored = DDP(restored, device_ids=[local_rank])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    assert restored_optimizer.state

    with torch.no_grad():
        actual = restored(inputs)
        expected = model(inputs)
    torch.testing.assert_close(actual, expected)

    dist.barrier()
    if rank == 0:
        os.remove(checkpoint_path)
        print("DDP checkpoint save/load round trip passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
