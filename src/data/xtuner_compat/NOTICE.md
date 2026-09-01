# XTuner compatibility code provenance

The pure-text soft-packing algorithm in `soft_pack.py` is maintained locally so
that cache construction does not depend on another checkout at runtime. Its
packing behavior was copied and reduced from:

`sgtm_v1/dataset/xtuner_vendor/datasets/packing.py`

The pretraining text preparation behavior in `pretrain.py` was copied and
reduced from:

`sgtm_v1/dataset/xtuner_vendor/datasets/pt_tokenize_fn/text.py`

Only the pure-text CPT behavior required by `xtuner_softpack_v1` is included.
Multimodal packing and dataset runtime code are intentionally out of scope.
