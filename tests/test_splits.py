from datasets.manifest import ManifestRecord
from datasets.splits import signer_independent_split


def test_signers_never_cross_splits():
    records = [
        ManifestRecord(sample_id=f"{signer}-{item}", dataset="demo", video_path="x.npy", signer_id=signer, label="a")
        for signer in ["a", "b", "c", "d", "e", "f"]
        for item in range(2)
    ]
    splits = signer_independent_split(records, seed=11)
    signer_sets = {split: {item.signer_id for item in values} for split, values in splits.items()}
    assert all(signer_sets.values())
    assert not signer_sets["train"] & signer_sets["val"]
    assert not signer_sets["train"] & signer_sets["test"]
    assert not signer_sets["val"] & signer_sets["test"]
