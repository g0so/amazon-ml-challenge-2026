# GPU CatBoost challenger

Native model uses the exact 22 packet features and positive class 1. The model was fit on `train/` only with CatBoost GPU, seed 2028, depth 6, learning rate 0.05, l2 leaf regularization 3, and early stopping against `tune/`.

Train/package command from the project root:

```powershell
& .\.venv\Scripts\python.exe scripts\train_gpu_challenger.py data\processed\gpu_feature_packet_v1_upload --output artifacts\gpu_challenger
```

CPU inference command:

```python
from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.load_model("model.cbm")
class1_probability = model.predict_proba(X)[:, 1]
prediction = class1_probability >= 0.362
```

Threshold selection uses official per-reference macro F0.5 on tuning only, including empty-reference coverage. The selected threshold is 0.362.
