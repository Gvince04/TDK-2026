# WESAD elofeldolgozo

- Uj szkript: tdk_framework/src/data/preprocess_wesad.py
- WESAD subject mappak (S2..S17) beolvasasa pickle-lel
- Chest ECG es EDA 60s ablak / 30s lepes, mean/std/min/max feature-ok
- Cimkek: 1 -> 0, 2 -> 1, tobbi eldobva
- Static feature: subjectenkent fix seed, 5 float
- Kimenet: cognitive_load_dataset.pt (CognitiveLoadDataset), azonos a preprocess_tdk.py kimenetevel

# WESAD kiértékelés

- run_all.py --dataset kapcsolo: wesad es tdk valaszthato
- WESAD nyers utvonal feloldas kulon fuggvenyben
- Eredmenyfajl datasetenkent kulon: experiment_results_wesad.json
- Egyosztalyos foldok kezelese: labels=[0, 1] a metrikakban, AUROC helyett nan
- sklearn figyelmeztetesek elnyomva a metrikaszamitasban
- WESAD pipeline lefutott: 859 minta, 8 dinamikus, 5 statikus feature
