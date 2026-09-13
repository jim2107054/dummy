# Bangladeshi Meme Classification with Qwen-VL + LoRA

Fine-tuning Vision-Language Models (Qwen2.5-VL / Qwen-VL) for 5-class Bangladeshi meme classification: **Political**, **Religious**, **Sports**, **Educational**, **Neutral**.

---

## Quickstart on Kaggle (Fetch from GitHub)

If your code is on GitHub (`https://github.com/jim2107054/dummy.git` or your repo), you can run everything on Kaggle in 3 simple steps:

### 1. Create a New Kaggle Notebook
1. Open Kaggle -> Create **New Notebook**.
2. Under **Notebook Settings** (right sidebar):
 - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
 - **Internet**: Switch to **On**.
3. Under **+ Add Input**, search and add dataset `mdjahidhasanjim/memedecode-train` (or attach your uploaded meme dataset).

### 2. Run the One-Liner Setup & Training Cell
```python
# 1. Clone repository from GitHub
!git clone https://github.com/jim2107054/dummy.git /kaggle/working/repo
%cd /kaggle/working/repo

# 2. Install / upgrade required packages
!pip install -q "transformers>=4.49.0" "peft>=0.13.0" "accelerate>=0.34.0" torchvision scikit-learn pillow

# 3. Start Training & Evaluation
# If dataset is uploaded as a Kaggle input or in repo:
!python train.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --epochs 5 --batch_size 1 --grad_accum 8 --data_dir /kaggle/input/datasets/mdjahidhasanjim/memedecode-train
```

---

## Running via Jupyter Notebook

You can directly upload and open [`bangla_meme_classification_kaggle.ipynb`](file:///d:/Dr.%20Sk.%20Md.%20Masudul%20Ahsan%20sir/intent_and_satire_of_bangla_meme/bangla_meme_classification_kaggle.ipynb) in Kaggle:

1. In Kaggle, click **File -> Upload Notebook** -> Select `bangla_meme_classification_kaggle.ipynb`.
2. Attach your dataset via **+ Add Input** -> Search for `mdjahidhasanjim/memedecode-train` (or upload dataset containing `train_image/Train/` and `labels.csv`).
3. Set Accelerator to **GPU T4 x2** (or **GPU P100**) and enable **Internet** (Turn ON).
4. Click **Run All**. All intermediate and final results will be plotted with rich interactive graphs (EDA distributions, sample galleries, training dynamics, confusion matrices, per-class metrics, t-SNE latent space, and misclassification error analysis)!

---

## Dataset Structure Supported

The pipeline automatically auto-detects both CSV-based and folder-based datasets under `/kaggle/input/datasets/mdjahidhasanjim/memedecode-train`, `/kaggle/input/memedecode-train`, `/kaggle/input/memedecode_train`, `/kaggle/input/MemeDecode_train`, `./`, or `./train_image`:

### Format A: CSV + Images (Default)
```
train_image/
├── labels.csv # Image_name, Label (Political, Religious, Sports, Educational, Neutral)
└── Train/ # train0001.jpg, train0002.jpg, ...
```

### Format B: Class Folders
```
dataset/
├── Political/ # *.jpg
├── Religious/ # *.jpg
├── Sports/ # *.jpg
├── Educational/ # *.jpg
└── Neutral/ # *.jpg
```

---

## Configuration & Hyperparameters

| Argument / Parameter | Default | Description |
|---|---|---|
| `--model_id` | `Qwen/Qwen2.5-VL-3B-Instruct` | Base vision-language model (e.g. 3B or 2B) |
| `--epochs` | `5` | Training epochs with early stopping on validation Macro-F1 |
| `--batch_size` | `1` | Batch size per GPU (expanded dynamically for evaluation) |
| `--grad_accum` | `8` | Gradient accumulation steps (effective batch size = 8) |
| `--lr` | `1e-4` | Learning rate with cosine schedule and warmup |
| `--max_pixels` | `256*28*28` | Image visual token budget to prevent OOM |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA scaling factor |

---

## Pipeline Overview

1. **Auto-Discovery**: Scans attached directories, matches `labels.csv` with images in `Train/`.
2. **Quality Check**: Decodes images, checks corrupted files, aspect ratios, and resolutions.
3. **Deduplication**: Computes SHA256 and 64-bit DCT perceptual hashes (pHash) to prevent data leakage.
4. **Stratified Split**: Groups duplicate components together and creates 70% Train, 15% Val, 15% Test splits.
5. **LoRA Fine-tuning**: Parameter-efficient training with generative teacher-forced loss on answer tokens only.
6. **Validation & Model Selection**: Evaluates Macro-F1 at the end of every epoch and checkpoints the best adapter.
7. **Test Evaluation**: Scores the held-out test split, outputs Confusion Matrix, Precision, Recall, Macro-F1, and saves all predictions to CSV.

---

## Output Artifacts

All results are saved to `/kaggle/working/meme_classification_results/`:
```
meme_classification_results/
├── model/ # Best LoRA adapter checkpoint (best_adapter)
├── splits/ # train.csv, val.csv, test.csv
├── metrics/ # test_metrics.json, classification_report.txt, confusion matrix
├── plots/ # Loss curves, class distributions, t-SNE / UMAP plots
└── predictions/ # test_predictions.csv (predictions + class probabilities)
```
