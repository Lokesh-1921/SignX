import numpy as np

from preprocessing.video import VideoLoader


def test_numpy_video_is_sampled_and_normalized(tmp_path):
    source = tmp_path / "clip.npy"
    np.save(source, np.full((3, 12, 16, 3), 255, dtype=np.uint8))
    frames = VideoLoader(num_frames=5, frame_size=(8, 10)).load(source)
    assert frames.shape == (5, 3, 8, 10)
    assert frames.min().item() == 1.0
    assert frames.max().item() == 1.0
