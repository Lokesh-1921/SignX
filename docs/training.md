# Training

Run a configured baseline with:

```bash
python train.py --config configs/experiment.yaml
```

The trainer selects CUDA first, then Apple MPS, then CPU. AMP is enabled only where supported. It records configuration, event metrics, a Git commit when available, optimizer state, latest/best/periodic checkpoints, and an append-only experiment registry entry. Each run gets a UUID-suffixed directory and existing runs are never overwritten.

For large datasets, use a manifest-backed DataLoader with modest workers, monitor host memory and decode throughput, and precompute visual embeddings only after a visual encoder version is fixed. Use `--resume` only with a checkpoint whose data vocabulary and configuration have been reviewed.
