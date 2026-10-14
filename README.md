# SignBridge

> Context-aware regional sign-language and gesture translation research platform.

SignBridge is a modular PyTorch research codebase for sign-language video understanding. It starts with a reproducible, signer-aware isolated-sign baseline and is designed to grow into pose fusion, continuous recognition, sign-to-text translation, and context-aware regional adaptation.

## Current milestone

Milestone 0 and the first runnable slice of Milestone 3 are implemented:

- universal JSONL manifests with validation and signer-independent splitting;
- lazy video/NumPy decoding, deterministic temporal sampling, and preprocessing metadata;
- RGB, pose, and RGB+pose temporal-transformer baseline;
- portable CUDA → Apple MPS → CPU device selection;
- checkpointing, structured JSONL experiment tracking, evaluation, and dataset validation;
- a synthetic end-to-end smoke run and unit tests.

No public dataset has been downloaded and no accuracy claim is made. The smoke run only verifies that the pipeline operates end to end.

## Quick start

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
pytest
python scripts/run_smoke_test.py
python train.py --config configs/experiment.yaml
```

`train.py` expects a manifest and source files specified in the configuration. Use the smoke test to generate a local, disposable example dataset.

## Layout

```text
datasets/       Dataset contracts, manifests, adapters, and splits
preprocessing/  Lazy decoding, sampling, and cached landmark contracts
models/         Visual, pose, temporal, fusion, VLM, and translation modules
training/       Device selection, trainer, checkpoints, experiment registry
evaluation/     Classification and sequence metrics
scripts/        Dataset validation, preparation, download planning, reproducibility
app/            FastAPI API foundation and frontend placeholder
docs/           Architecture, dataset, ethics, and operating documentation
```

## Responsible use

Sign languages are natural languages. Regional and informal variation is legitimate, not an error category. Predictions are probabilistic and must not be represented as guaranteed translations. Custom recording requires informed consent; no regional meanings or model results are fabricated by this repository.

## Large-scale dataset stage

After reviewing a dataset's licence and access process, first create/verify its adapter and run:

```bash
python scripts/download_dataset.py --dataset wlasl
python scripts/prepare_dataset.py --dataset wlasl --workers 8 --shard 0/20
python scripts/validate_dataset.py --manifest data/manifests/wlasl/train.jsonl
```

The first command only reports the transfer plan. After you have confirmed projected disk use, licence, and an official access URL, the explicit transfer form is:

```bash
python scripts/download_dataset.py --dataset wlasl --source-url "<official-authorized-url>" --sha256 "<official-checksum>" --confirm
```

See [docs/architecture.md](docs/architecture.md), [docs/datasets.md](docs/datasets.md), and [CHANGELOG.md](CHANGELOG.md).
