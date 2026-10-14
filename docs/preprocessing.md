# Preprocessing

`VideoLoader` decodes a sample only when a DataLoader requests it. It accepts regular video files through OpenCV and `.npy`/`.npz` arrays for testing/cached clips, samples a fixed temporal window, converts to RGB tensors, normalizes to `[0, 1]`, and resizes on demand.

The scalable processing plan is validate → inspect metadata → decode → normalize FPS → crop/resize → landmark extraction → temporal clip creation → write manifest. Process samples in named shards (`--shard 0/20`) and cache expensive landmarks under `data/landmarks/`; never repeatedly infer them in every epoch.

`LandmarkSequence` has left/right hand, body pose, optional face, and optional confidence arrays. It is extractor-neutral: extraction tooling must declare its version, coordinate convention, and confidence handling before its cache is used for training.

Augmentations belong in configuration. Horizontal flips and any spatial transformation must be enabled only after linguistic review for the target language/task, because they can change sign meaning.
