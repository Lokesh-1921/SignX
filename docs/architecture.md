# Architecture

SignBridge uses contracts rather than a fixed single model:

```text
dataset adapter → universal manifest → lazy video/landmark loader
                                   ↓
       VisualEncoder + PoseEncoder → Fusion → TemporalTransformer
                                              ↓
              sign embedding → classifier / CTC decoder / text decoder
```

The working Milestone 0–3 path is `TinyFrameEncoder → TemporalTransformer → classifier`. `SignBridgeBaseline` exposes a reusable sign embedding and accepts an optional landmark tensor, so RGB-only, pose-only (with a future pose-only wrapper), and RGB+pose fusion can be compared using the same evaluation protocol.

Future video encoders (VideoMAE, Video Swin, TimeSformer) implement `VisualEncoder`; future VLMs implement `VisionLanguageModel`. Continuous CTC recognition and sign-to-text are deliberately separate heads, not labels fabricated from isolated-sign data.

All sequence components accept an attention mask. Fixed temporal windows are used for the current baseline; variable-length collators and CTC decoding are the next recognition milestone.
