# Pixels to Predictions: DL Vision Challenge

**Course:** Deep Learning (CSGY 6953), New York University  
**Instructor:** Gustavo Sandoval  
**Competition:** [Pixels to Predictions: DL Vision Challenge](https://www.kaggle.com/competitions/pixels-to-predictions) on Kaggle  
**Final Score:** 0.48692 (48.69% accuracy) | Rank: 104/105  
**Author:** Anubhuti Mathur (am15443@nyu.edu)

---

## Overview

This repository contains all notebooks, the trained LoRA adapter, and documentation for my submission to the Pixels to Predictions competition. The task is multimodal science multiple-choice question answering: given an image and question context, select the correct answer from 2–5 candidates.

**Approach:** QLoRA fine-tuning of `HuggingFaceTB/SmolVLM-500M-Instruct` (≤5M trainable parameters) with log-likelihood scoring for inference.

---

## Repository Structure

```
pixels-to-predictions/
│
├── notebooks/
│   ├── notebook_A_baseline.ipynb          # Baseline: r=16, q+v, 3 epochs, train only
│   ├── notebook_B_improved.ipynb          # Best run: r=16, q+k+v+o, 5 epochs, train+val
│   ├── notebook_B_inference_only.ipynb    # Inference-only for saved adapter (img=224)
│   ├── inference_loglik_448.ipynb         # Inference-only with img=448 (final submission)
│   ├── notebook_C_lr5e4.ipynb             # Experiment: lr=5e-4 (not submitted)
│   └── notebook_E_nosolution.ipynb        # Experiment: no solution field (incomplete)
│
├── adapter/                               # Trained LoRA adapter weights (Notebook B)
│   ├── adapter_model.safetensors          # Trained weights (16.7 MB)
│   ├── adapter_config.json
│   ├── tokenizer.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── tokenizer_config.json
│   ├── preprocessor_config.json
│   ├── processor_config.json
│   ├── special_tokens_map.json
│   ├── added_tokens.json
│   └── chat_template.jinja
│
├── .gitignore
└── README.md
```

---

## Competition Constraints

| Constraint | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolVLM-500M-Instruct` only |
| Max trainable parameters | 5,000,000 |
| Allowed data | Competition data only |
| Compute | Kaggle Free Tier (45 GPU hours/week) |
| Evaluation | Offline (no internet at inference time) |

---

## Results Summary

| Notebook | LoRA Rank | Modules | Epochs | Training Data | Solution Field | Score |
|---|---|---|---|---|---|---|
| A (baseline) | 16 | q, v | 3 | train only | no | 39.24% |
| B (improved) | 16 | q, k, v, o | 5 | train + val | yes (200 chars) | 48.49% |
| **B-448 (final)** | **16** | **q, k, v, o** | **5** | **train + val** | **yes** | **48.69%** |
| C (lr=5e-4) | 16 | q, k, v, o | 3 | train + val | yes | not submitted |
| E (no solution) | 16 | q, k, v, o | 3* | train + val | no | not completed |

*Notebook E did not finish before the deadline.  
B-448 uses the same trained adapter as B with `IMG_SIZE=448` at inference only.

---

## How to Run

### Prerequisites

- A [Kaggle account](https://kaggle.com) with phone verification (required for submission)
- Access to the [competition page](https://www.kaggle.com/competitions/pixels-to-predictions) — join and accept the rules
- GPU T4 x2 enabled on your Kaggle notebook (right sidebar → Accelerator)

---

### Step 1: Set Up the Kaggle Notebook

1. Go to the competition page → **Code** tab → **New Notebook**
2. This automatically mounts the competition data at `/kaggle/input/competitions/pixels-to-predictions/`
3. Upload the notebook you want to run: **File → Import notebook**
4. In the right sidebar, set **Accelerator → GPU T4 x2**

---

### Step 2: Verify Data Paths

The competition data is mounted at:
```
/kaggle/input/competitions/pixels-to-predictions/
├── train.csv
├── val.csv
├── test.csv
├── sample_submission.csv
└── images/
    └── images/
        ├── train/   (3,109 images)
        ├── val/     (1,048 images)
        └── test/    (1,008 images)
```

Note the double-nested `images/images/` path — this is handled automatically in all notebooks via:
```python
fixed_path = str(rel_path).replace("images/", "images/images/", 1)
```

Confirm paths are correct by running this in a cell:
```python
import os
for root, dirs, files in os.walk("/kaggle/input/competitions/pixels-to-predictions/images/images"):
    print(f"{root}: {len(files)} files")
```

---

### Step 3: Run a Training Notebook (e.g., Notebook B)

Open `notebook_B_improved.ipynb`. The key configuration is in the first code cell:

```python
DATA_DIR    = Path("/kaggle/input/competitions/pixels-to-predictions")
OUTPUT_DIR  = Path("/kaggle/working/outputs")
MODEL_ID    = "HuggingFaceTB/SmolVLM-500M-Instruct"

IMG_SIZE      = 224
LORA_R        = 16
LORA_ALPHA    = 32
NUM_EPOCHS    = 5
LEARNING_RATE = 2e-4
MAX_SEQ_LEN   = 2048
```

No changes needed — just click **Save Version → Save & Run All**.

**Expected runtime:** ~12 hours (5 epochs × ~2.4 hours each).

> ⚠️ **Important:** Kaggle has a 12-hour maximum execution time per notebook session. Notebook B hits this limit during inference. Use the inference-only notebook (Step 5) to complete predictions from the saved adapter.

---

### Step 4: Upload Adapter as a Kaggle Dataset

The adapter saved during training needs to be available for the inference notebook. Either:

**Option A — Use the adapter from this repo:**
Upload the `adapter/` folder from this repo to a new Kaggle Dataset named `smolvlm-finetuned-b`.

**Option B — Use the adapter saved during your own training run:**
After training completes, download the files from `/kaggle/working/outputs/lora_adapter/` in the notebook's Output tab, then upload them as a Kaggle Dataset.

To create the dataset:
1. Go to kaggle.com → your profile → **Your Work → Datasets → New Dataset**
2. Upload all adapter files
3. Name it `smolvlm-finetuned-b`, set to **Private**, click **Create**

---

### Step 5: Run Inference

Upload `inference_loglik_448.ipynb` (final submission) or `notebook_B_inference_only.ipynb` (img=224) to Kaggle as a new notebook.

**Add the adapter dataset:**
- Right sidebar → **Add Data** → search for `smolvlm-finetuned-b` → Add

**Verify the adapter path** by running in a cell:
```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        print(os.path.join(root, f))
    if files:
        break
```

Update `ADAPTER_DIR` in the config cell to match:
```python
ADAPTER_DIR = Path("/kaggle/input/datasets/YOUR_USERNAME/smolvlm-finetuned-b/outputs/lora_adapter")
```

Enable **GPU T4 x2** and click **Save & Run All**.

**Expected runtime:** ~1 hour for 1,008 test examples.

---

### Step 6: Submit Predictions

1. After the inference notebook completes, go to its **Output** tab
2. Download `submission.csv`
3. Go to the competition page → **Submit Predictions** → upload the file
4. Your score appears on the leaderboard within a few minutes

---

## Running on Google Colab

Use Colab to run experiments in parallel with Kaggle (Kaggle limits to 2 concurrent GPU sessions).

### Setup

```python
# 1. Mount Google Drive to save outputs across sessions
from google.colab import drive
drive.mount('/content/drive')

# 2. Download competition data via Kaggle API
!pip install kaggle
from google.colab import files
files.upload()  # upload kaggle.json when prompted
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c pixels-to-predictions
!unzip -q pixels-to-predictions.zip -d /content/

# 3. Update paths in notebook config cell:
DATA_DIR   = Path("/content")
OUTPUT_DIR = Path("/content/drive/MyDrive/notebook_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
```

### Keeping Colab Alive

Colab disconnects after ~90 minutes of inactivity. Run this in your browser console (F12 → Console):

```javascript
function KeepAlive() {
  document.querySelector('#top-toolbar').click();
  console.log('Kept alive: ' + new Date().toLocaleTimeString());
}
setInterval(KeepAlive, 60000);
```

### Path differences from Kaggle

| Item | Kaggle | Colab |
|---|---|---|
| Data directory | `/kaggle/input/competitions/pixels-to-predictions` | `/content` |
| Output directory | `/kaggle/working/outputs` | `/content/outputs` |
| Submission file | `/kaggle/working/submission.csv` | `/content/submission.csv` |

---

## Key Technical Notes

### Trainable Parameter Count

With `LORA_R=16` targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`:
```
trainable params: 4,161,536 || all params: 511,643,840 || trainable%: 0.8134
```
Safely under the 5,000,000 parameter competition cap.

### Image Path Fix

The competition data has a double-nested image directory (`images/images/train/`). All notebooks handle this automatically:
```python
def _load_image(self, rel_path):
    fixed_path = str(rel_path).replace("images/", "images/images/", 1)
    return Image.open(self.data_dir / fixed_path).convert("RGB")...
```

### Offline Evaluation

The competition evaluates in an offline environment without internet. The base model must be downloaded during the notebook run (Kaggle caches HuggingFace models between sessions). To guarantee offline reproducibility, save the base model locally after loading:
```python
base_model.save_pretrained("/kaggle/working/smolvlm_base")
processor.save_pretrained("/kaggle/working/smolvlm_base")
```

---

## Dependencies

```
transformers==4.57.6
peft==0.18.1
bitsandbytes
accelerate
torch
pillow
pandas
numpy
tqdm
```

Install with:
```bash
pip install transformers==4.57.6 peft==0.18.1 bitsandbytes accelerate pillow tqdm
```
