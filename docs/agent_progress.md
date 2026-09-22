# WESAD elofeldolgozo

- Uj szkript: tdk_framework/src/data/preprocess_wesad.py
- WESAD subject mappak (S2..S17) beolvasasa pickle-lel
- Chest ECG es EDA 60s ablak / 30s lepes, mean/std/min/max feature-ok
- Cimkek: 1 -> 0, 2 -> 1, tobbi eldobva
- Static feature: subjectenkent fix seed, 5 float
- Kimenet: cognitive_load_dataset.pt (CognitiveLoadDataset), azonos a preprocess_tdk.py kimenetevel
