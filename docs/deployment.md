# Deployment

The current FastAPI foundation offers `/health` and `/models`. Prediction, translation, upload, and streaming must remain unavailable until a validated checkpoint is registered with a model card and input/output contract.

The real-time serving architecture is webcam → bounded frame buffer → sliding temporal window → preprocessing → model → debouncing/smoothing → labelled confidence and latency. It must avoid independently processing every frame and show predictions as uncertain. Production deployments require size limits, authentication, rate limits, private-upload retention/deletion policy, structured logs without raw personal video, and hardware-specific benchmarking.
