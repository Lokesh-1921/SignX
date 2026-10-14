# Experiments

`experiments/runs/<experiment-id>/` stores the resolved configuration and JSONL events. `experiments/registry.json` indexes completed local runs. Figures, confusion matrices, embeddings, and latency reports belong under that run's `figures/` folder.

Planned ablations are RGB only, pose only, RGB+pose, RGB+pose+face; with/without temporal attention; with/without contextual input; and regional fine-tuning comparison. They become valid only after source data, split protocol, seed, checkpoint, hardware, and raw results are recorded.
