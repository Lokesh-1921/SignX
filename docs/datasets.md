# Dataset policy and manifests

Every dataset is represented as JSONL, one `ManifestRecord` per sample. Required fields are `sample_id`, `dataset`, and `video_path`; important optional fields include signer ID, label/gloss, translation, language, region, duration/FPS/resolution, split, and landmark path. Dataset-specific annotations stay in `extra` so provenance is not lost.

Before adding data, document the official source, exact release, license/access conditions, language, annotation schema, sample/class counts, signer and regional metadata, and conversion commit in `docs/datasets/`. Do not merge label spaces blindly. Use source-specific adapters to map into the universal schema.

For generalization experiments, run `signer_independent_split` only when signer IDs are present. Never place a signer in multiple splits. Cross-region and cross-dataset protocols must be recorded in the experiment configuration.

The repository holds neither raw datasets nor private recordings. `scripts/download_dataset.py` plans transfers first and requires an explicit source URL and `--confirm` to transfer anything.
