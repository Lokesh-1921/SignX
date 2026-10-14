from datasets.manifest import ManifestError, ManifestRecord, iter_manifest, write_manifest
from scripts.validate_dataset import validate_manifest


def test_round_trip_preserves_unknown_provenance(tmp_path):
    record = ManifestRecord.from_dict({"sample_id": "x1", "dataset": "demo", "video_path": "clip.npy", "release_id": "2026-a"})
    path = tmp_path / "records.jsonl"
    write_manifest([record], path)
    loaded = list(iter_manifest(path))
    assert loaded == [record]
    assert loaded[0].extra == {"release_id": "2026-a"}


def test_manifest_requires_nonempty_identifiers():
    try:
        ManifestRecord(sample_id="", dataset="demo", video_path="clip.npy")
    except ManifestError as error:
        assert "sample_id" in str(error)
    else:
        raise AssertionError("empty sample ID must fail validation")


def test_dataset_validator_checks_local_source(tmp_path):
    source = tmp_path / "clip.npy"
    source.write_bytes(b"placeholder")
    manifest = tmp_path / "records.jsonl"
    write_manifest([ManifestRecord(sample_id="x", dataset="demo", video_path=str(source), label="label", signer_id="person")], manifest)
    assert validate_manifest(manifest) == {"samples": 1, "labels": 1, "signers": 1}
