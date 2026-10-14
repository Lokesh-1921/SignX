import torch

from training.checkpoints import CheckpointManager


def test_checkpoint_round_trip(tmp_path):
    manager = CheckpointManager(tmp_path / "run")
    path = manager.save("latest.pt", {"epoch": 3, "tensor": torch.tensor([4])})
    payload = manager.load(path, torch.device("cpu"))
    assert payload["epoch"] == 3
    assert payload["tensor"].item() == 4
