# Native MANO research heads

| File | Role | Scope |
|---|---|---|
| `v7_full.pt` | Native MANO DiT, expanded natural training data | Right-only, fixed RGB backbone, wrist and shape |
| `v7_small.pt` | Matched small-data control | Same native architecture and update budget |
| `v9_spatial.pt` | Independent RGB heatmap/risk head | Full 16×12 spatial features |
| `v9_pooled.pt` | Matched pooled-feature control | Same trainable architecture |

These four files are regular Git blobs (each under 10 MB), so they do not depend
on the account's exhausted LFS budget. Historical checkpoints elsewhere still use
LFS. Sizes, SHA-256 and original training-checkpoint hashes
are in `evidence/native_mano_h20_20261019/manifest.json`. The v7 files are stripped
inference exports; their model tensors exactly match the selected training files.
They do not provide complete optimizer/RNG/data state for training resumption.

No WiLoR backbone, MANO asset, raw data or GT is included. None of these research
heads is admitted as a production replacement. See `docs/NATIVE_MANO_H20.md` for
native parameter semantics, dependencies, results and limitations.
