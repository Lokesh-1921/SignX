import torch

from models import SignBridgeBaseline


def test_rgb_baseline_forward_shape():
    model = SignBridgeBaseline(num_classes=3, hidden_dim=16, temporal_layers=1, temporal_heads=4)
    output = model(torch.rand(2, 4, 3, 32, 32), torch.ones(2, 4, dtype=torch.bool))
    assert output["logits"].shape == (2, 3)
    assert output["embedding"].shape == (2, 16)


def test_rgb_pose_baseline_forward_shape():
    model = SignBridgeBaseline(num_classes=2, hidden_dim=16, temporal_layers=1, temporal_heads=4, use_pose=True, pose_dim=6)
    output = model(torch.rand(2, 4, 3, 32, 32), torch.ones(2, 4, dtype=torch.bool), torch.rand(2, 4, 6))
    assert output["logits"].shape == (2, 2)
