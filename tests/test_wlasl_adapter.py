"""Tests for the WLASL dataset adapter."""

import json

import pytest

from datasets.adapters import WLASLAdapter
from datasets.base import adapter_registry


def _make_wlasl_annotation(root, glosses, videos_per_gloss=2):
    """Create a minimal WLASL annotation JSON and dummy video files."""
    annotation = []
    video_dir = root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    video_counter = 0
    for gloss in glosses:
        instances = []
        for i in range(videos_per_gloss):
            video_id = f"{video_counter:05d}"
            video_counter += 1
            # Create a dummy video file (just needs to exist)
            (video_dir / f"{video_id}.mp4").write_bytes(b"fake")
            split = "train" if i % 2 == 0 else "val"
            instances.append({
                "video_id": video_id,
                "signer_id": i,
                "source": "test",
                "split": split,
                "bbox": [0, 0, 100, 100],
                "frame_start": 0,
                "frame_end": -1,
                "url": "http://example.com",
            })
        annotation.append({"gloss": gloss, "instances": instances})

    annotation_path = root / "WLASL_v0.3.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    return annotation_path


def test_wlasl_adapter_registered():
    """WLASL adapter should be in the global registry."""
    assert "wlasl" in adapter_registry


def test_wlasl_adapter_parses_annotation(tmp_path):
    """Adapter should produce ManifestRecords from a WLASL JSON fixture."""
    _make_wlasl_annotation(tmp_path, ["hello", "world", "book"], videos_per_gloss=3)
    adapter = WLASLAdapter()
    records = list(adapter.build_records(tmp_path))
    assert len(records) == 9  # 3 glosses × 3 videos
    labels = {r.label for r in records}
    assert labels == {"hello", "world", "book"}
    assert all(r.dataset == "wlasl" for r in records)
    assert all(r.language == "ASL" for r in records)


def test_wlasl_adapter_subset_filter(tmp_path):
    """Subset parameter should limit glosses by index."""
    _make_wlasl_annotation(tmp_path, ["a", "b", "c", "d", "e"], videos_per_gloss=2)
    adapter = WLASLAdapter(subset=3)
    records = list(adapter.build_records(tmp_path))
    labels = {r.label for r in records}
    assert labels == {"a", "b", "c"}
    assert "d" not in labels
    assert "e" not in labels


def test_wlasl_adapter_preserves_provenance(tmp_path):
    """Extra fields (bbox, source, etc.) should be preserved in record.extra."""
    _make_wlasl_annotation(tmp_path, ["test"], videos_per_gloss=1)
    adapter = WLASLAdapter()
    records = list(adapter.build_records(tmp_path))
    assert len(records) == 1
    record = records[0]
    assert "bbox" in record.extra
    assert "source" in record.extra
    assert record.extra["gloss_index"] == 0


def test_wlasl_adapter_skips_missing_videos(tmp_path):
    """Adapter should skip videos that don't exist and warn."""
    annotation = [{"gloss": "test", "instances": [
        {"video_id": "99999", "signer_id": 0, "split": "train"},  # doesn't exist
    ]}]
    # Also create one valid video to avoid the "no records" error
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "00001.mp4").write_bytes(b"fake")
    annotation.append({"gloss": "valid", "instances": [
        {"video_id": "00001", "signer_id": 1, "split": "train"},
    ]})
    (tmp_path / "WLASL_v0.3.json").write_text(json.dumps(annotation))
    adapter = WLASLAdapter()
    records = list(adapter.build_records(tmp_path))
    assert len(records) == 1
    assert records[0].label == "valid"


def test_wlasl_adapter_signer_ids(tmp_path):
    """Signer IDs should be formatted consistently."""
    _make_wlasl_annotation(tmp_path, ["hello"], videos_per_gloss=3)
    adapter = WLASLAdapter()
    records = list(adapter.build_records(tmp_path))
    signer_ids = {r.signer_id for r in records}
    assert all(s.startswith("wlasl_signer_") for s in signer_ids)


def test_wlasl_adapter_no_annotation_raises(tmp_path):
    """Missing annotation file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Could not find WLASL"):
        list(WLASLAdapter().build_records(tmp_path))
