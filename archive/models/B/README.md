# Repaired-retrieval CatBoost model

The model uses the exact ordered 22-feature V3 schema. TRAIN alone fits the model; TUNE drives early stopping and threshold calibration; CONFIRMATION is evaluated once at the locked threshold. The frozen baseline is calibrated on TUNE and compared on the identical CONFIRMATION candidates.

Reproduce from the project root with `python scripts/train_repaired_packet.py path/to/repaired_feature_packet.zip`. The ZIP is hash-checked, extracted to a packet-SHA-addressed directory, and all partition offsets, candidate rows, truth rows, labels, hashes and split disjointness are verified before fitting.
