# Bangladeshi Meme Classification with Qwen3-VL + LoRA

Research notebook for five-class meme classification: Political, Religious, Sports, Educational,
Harmless.

`qwen3vl_lora_bangla_meme_classification.ipynb` is a self-contained Kaggle notebook covering
dataset discovery, quality and duplicate analysis, a leakage-safe stratified split, LoRA
fine-tuning of Qwen3-VL, a single held-out test evaluation, and the feature-space and error
analyses.

## Running it on Kaggle

1. Upload the notebook, attach your meme dataset, and set the accelerator to **GPU T4 x2** or
   **P100**. Turn **Internet** on unless you also attach the model weights as a dataset.
2. Run the cells in order. Sections 01 to 10 do all the checks and print a configuration summary
   before anything expensive starts, so a misconfigured run can be stopped in the first minute.
3. Section 11 trains. Everything after it reads the artifacts that section produced.

The dataset path is discovered at runtime, not hard-coded. Any folder layout that puts the five
classes in their own directories works, including one extra nesting level or an existing
train/val/test split, which is merged and re-split so the duplicate-aware logic applies to the
whole corpus. If a class folder is named something the notebook does not recognise, add it to
`CONFIG["class_aliases"]` in Section 01 and re-run Section 03.

## Configuration worth knowing about

Everything tunable is in the `CONFIG` dictionary in Section 01.

| Setting | Default | When to change it |
|---|---|---|
| `model_id` | `Qwen/Qwen3-VL-2B-Instruct` | Larger variants need more VRAM. The 4B model fits a single T4 only with a smaller `max_pixels`. |
| `max_pixels` | `256*28*28` | The main lever against CUDA out-of-memory. Lower it first when training will not fit. |
| `near_dup_hamming_max` | `5` | Lower it to 3 or 2 if Section 05 warns that one duplicate component swallowed a large share of the corpus. |
| `aug.hflip_p` | `0.0` | Horizontal flip is off because mirrored Bangla and English captions become unreadable. Raise it only for a text-free corpus. |
| `train.epochs`, `lr`, `grad_accum_steps` | 3, 1e-4, 8 | Standard LoRA starting point. Tune against validation only. |

## Outputs

Written to `/kaggle/working/meme_classification_results/`:

```
model/        LoRA adapter (best and final), processor, adapter card
splits/       dataset index and train/val/test CSVs, unreadable-file list
metrics/      test metrics, classification report, confusion matrices, separability,
              class-centroid similarity, per-class analysis, training history, config
plots/        every figure as PNG at 200 dpi
predictions/  per-image predictions with probabilities, misclassifications, confident errors
embeddings/   test and validation embeddings, t-SNE and UMAP coordinates
```

## Method notes for the write-up

* The model stays generative. It is trained to emit one class name, with the loss applied to the
  answer tokens only. At inference the five class strings are scored and renormalised, which gives
  a probability over exactly the five classes rather than free-form text.
* Duplicate and near-duplicate images are grouped into connected components, and whole components
  are assigned to a single split. The split cell asserts that no component, byte-identical file or
  path is shared between splits before training starts.
* The test split is read once, in Section 13, after model selection on validation macro-F1.
* t-SNE, UMAP and the centroid-similarity matrix are exploratory. The quantitative separability
  numbers are computed in the original high-dimensional space, and the notebook states this
  distinction where each result is reported.
