# ============================================================
# PASTE THIS AS A NEW CELL RIGHT AFTER trainer.train() FINISHES
# (and again after Step 11's clause_embeddings are built)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os, json, numpy as np

DRIVE_ROOT = "/content/drive/MyDrive/Beacon_Legal/artifacts/cuad"
os.makedirs(DRIVE_ROOT, exist_ok=True)

# 1. Save the fine-tuned QA model + tokenizer straight to Drive (skip the
#    /content copy step entirely - write here directly so a dropped
#    runtime can never lose it)
MODEL_DRIVE_PATH = f"{DRIVE_ROOT}/legalbert_final"
trainer.save_model(MODEL_DRIVE_PATH)
tokenizer.save_pretrained(MODEL_DRIVE_PATH)
print("Model saved to:", MODEL_DRIVE_PATH)

# 2. Save the retrieval index (embeddings + the clause records they map to)
#    Run this part after Step 11 has built clause_embeddings / cuad_clause_records.
if "clause_embeddings" in globals():
    np.save(f"{DRIVE_ROOT}/clause_embeddings.npy", clause_embeddings)
    with open(f"{DRIVE_ROOT}/clause_records.json", "w", encoding="utf-8") as f:
        json.dump(cuad_clause_records, f)
    print("Retrieval index saved to:", DRIVE_ROOT)
else:
    print("clause_embeddings not built yet - re-run this cell after Step 11.")

# 3. (Optional) also pull a zip to your local machine instead of/as well as Drive
# !zip -r /content/cuad_artifacts.zip "$MODEL_DRIVE_PATH" \
#     "{DRIVE_ROOT}/clause_embeddings.npy" "{DRIVE_ROOT}/clause_records.json"
# from google.colab import files
# files.download("/content/cuad_artifacts.zip")
