"""Dataset contracts, adapters, and manifest-backed PyTorch datasets."""

from .base import DatasetAdapter, DatasetMetadata, adapter_registry
from .manifest import ManifestError, ManifestRecord, iter_manifest, write_manifest
from .video_dataset import SignVideoDataset

__all__ = [
    "DatasetAdapter",
    "DatasetMetadata",
    "ManifestError",
    "ManifestRecord",
    "SignVideoDataset",
    "adapter_registry",
    "iter_manifest",
    "write_manifest",
]
