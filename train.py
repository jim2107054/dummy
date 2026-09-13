#!/usr/bin/env python3
"""
Bangladeshi Meme Classification with Qwen-VL + LoRA
End-to-end Pipeline: Dataset Discovery -> Quality Check -> Deduplication -> Stratified Split -> LoRA Training -> Evaluation -> Error Analysis -> Export
"""

import os, sys, json, math, random, hashlib, warnings, time, argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ----------------- Command Line Arguments & Config -----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen-VL LoRA for Bangla Meme Classification")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Hugging Face Model ID or local path")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_pixels", type=int, default=256*28*28, help="Max image pixel budget")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/meme_classification_results", help="Output directory")
    parser.add_argument("--data_dir", type=str, default=None, help="Explicit data directory or search root")
    return parser.parse_args()

args = parse_args()

CONFIG = {
    "experiment_name": "qwen_vl_lora_bd_meme_5class",
    "seed": args.seed,
    "classes": ["Political", "Religious", "Sports", "Educational", "Neutral"],
    "search_roots": [args.data_dir] if args.data_dir else ["/kaggle/input", "/kaggle/working/data", "/kaggle/working", "./train_image", "./data", "."],
    "image_extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"],
    "class_aliases": {
        "political": "Political", "politics": "Political", "politic": "Political",
        "politicalmemes": "Political", "politicalmeme": "Political",
        "religious": "Religious", "religion": "Religious", "religiousmemes": "Religious",
        "sports": "Sports", "sport": "Sports", "sportsmemes": "Sports", "sportsmeme": "Sports",
        "educational": "Educational", "education": "Educational", "educationalmemes": "Educational",
        "harmless": "Neutral", "neutral": "Neutral", "harmlessmemes": "Neutral",
        "nonharmful": "Neutral", "nonharmless": "Neutral",
    },
    "phash_size": 8,
    "near_dup_hamming_max": 5,
    "run_near_dup": True,
    "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    "model_id": args.model_id,
    "attn_implementation": "sdpa",
    "min_pixels": 64 * 28 * 28,
    "max_pixels": args.max_pixels,
    "lora": {
        "r": args.lora_r, "alpha": args.lora_alpha, "dropout": 0.05, "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "adapt_vision_tower": False,
    },
    "train": {
        "epochs": args.epochs,
        "per_device_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "lr": args.lr,
        "weight_decay": 0.0,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "early_stopping_patience": 2,
        "eval_batch_images": 2,
        "log_every_steps": 10,
        "max_train_samples": None,
        "max_eval_samples": None,
    },
    "aug": {
        "hflip_p": 0.0,
        "rotation_deg": 5.0, "rotation_p": 0.30,
        "brightness": 0.12, "contrast": 0.12, "saturation": 0.08, "jitter_p": 0.30,
        "random_resized_crop_p": 0.25, "crop_scale": (0.88, 1.0),
    },
    "out_dir": args.out_dir if os.path.exists("/kaggle/working") else "./meme_classification_results",
}

CLASSES = CONFIG["classes"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}
CLASS_COLORS = {"Political": "#4C72B0", "Religious": "#DD8452", "Sports": "#55A868",
                "Educational": "#C44E52", "Neutral": "#8172B3", "Harmless": "#8172B3"}

OUT = Path(CONFIG["out_dir"])
for sub in ["model", "splits", "metrics", "plots", "predictions", "embeddings", "logs"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

print(f"=== Experiment: {CONFIG['experiment_name']} ===")
print(f"Classes   : {CLASSES}")
print(f"Output dir: {OUT.resolve()}")

# ----------------- Reproducibility -----------------
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = CONFIG["seed"]
set_seed(SEED)

# ----------------- Dataset Discovery -----------------
IMG_EXT = set(CONFIG["image_extensions"])

def normalise_str(name: str) -> str:
    s = str(name).strip().lower()
    for ch in [" ", "_", "-", ".", "(", ")"]:
        s = s.replace(ch, "")
    return s

def canonical_class(label_name: str):
    if label_name is None or pd.isna(label_name):
        return None
    key = normalise_str(label_name)
    if key in CONFIG["class_aliases"]:
        return CONFIG["class_aliases"][key]
    for c in CLASSES:
        if key == normalise_str(c) or normalise_str(c) in key:
            return c
    return None

def count_images_in(d: Path) -> int:
    n = 0
    try:
        for e in os.scandir(d):
            if e.is_file() and Path(e.name).suffix.lower() in IMG_EXT:
                n += 1
    except (PermissionError, OSError):
        pass
    return n

def index_all_images(roots, max_depth=4):
    img_map = {}
    for r in roots:
        if not r: continue
        p = Path(r)
        if not p.exists(): continue
        for dirpath, dirnames, filenames in os.walk(p):
            depth = len(Path(dirpath).relative_to(p).parts)
            if depth > max_depth or ".git" in dirpath or ".ipynb_checkpoints" in dirpath:
                dirnames[:] = []
                continue
            for fn in filenames:
                if Path(fn).suffix.lower() in IMG_EXT:
                    img_map[fn.lower()] = str((Path(dirpath) / fn).resolve())
    return img_map

def discover_csv_records(roots, img_map, max_depth=4):
    for r in roots:
        if not r: continue
        p = Path(r)
        if not p.exists(): continue
        for dirpath, dirnames, filenames in os.walk(p):
            depth = len(Path(dirpath).relative_to(p).parts)
            if depth > max_depth or ".git" in dirpath:
                dirnames[:] = []
                continue
            for fn in filenames:
                if fn.lower().endswith(".csv") and not fn.startswith("submission"):
                    csv_path = Path(dirpath) / fn
                    try:
                        df = pd.read_csv(csv_path)
                    except Exception:
                        continue
                    img_col, lbl_col = None, None
                    for col in df.columns:
                        cn = normalise_str(col)
                        if cn in ["imagename", "image", "filename", "file", "id", "img", "imagepath", "filepath"]:
                            img_col = col
                            break
                    for col in df.columns:
                        cn = normalise_str(col)
                        if cn in ["label", "category", "class", "target", "intent", "satire"]:
                            lbl_col = col
                            break
                    if img_col and lbl_col:
                        records = []
                        for _, row in df.iterrows():
                            img_name = str(row[img_col]).strip()
                            canon = canonical_class(row[lbl_col])
                            if not canon: continue
                            img_file = None
                            for sub in ["", "Train", "train", "images", "Images"]:
                                cand = Path(dirpath) / sub / img_name if sub else Path(dirpath) / img_name
                                if cand.exists():
                                    img_file = str(cand.resolve())
                                    break
                            if not img_file and img_name.lower() in img_map:
                                img_file = img_map[img_name.lower()]
                            if img_file and Path(img_file).exists():
                                records.append({
                                    "filepath": img_file, "filename": Path(img_file).name,
                                    "class": canon, "label_id": CLASS_TO_ID[canon],
                                    "source_dir": str(Path(img_file).parent),
                                    "extension": Path(img_file).suffix.lower()
                                })
                        if len(records) > 0:
                            print(f"[CSV Match] Loaded {len(records)} samples from {csv_path}")
                            return records
    return None

def discover_folder_records(roots, max_depth=4):
    found = defaultdict(list)
    for r in roots:
        if not r: continue
        p = Path(r)
        if not p.exists(): continue
        for dirpath, dirnames, filenames in os.walk(p):
            depth = len(Path(dirpath).relative_to(p).parts)
            if depth > max_depth or ".git" in dirpath:
                dirnames[:] = []
                continue
            cls = canonical_class(Path(dirpath).name)
            if cls is not None and count_images_in(Path(dirpath)) > 0:
                found[cls].append(Path(dirpath))
    records = []
    seen = set()
    for cls, dirs in found.items():
        for d in dirs:
            for e in sorted(os.scandir(d), key=lambda x: x.name):
                if not e.is_file(): continue
                p = Path(e.path)
                if p.suffix.lower() not in IMG_EXT: continue
                rp = str(p.resolve())
                if rp in seen: continue
                seen.add(rp)
                records.append({
                    "filepath": rp, "filename": p.name, "class": cls,
                    "label_id": CLASS_TO_ID[cls], "source_dir": str(d),
                    "extension": p.suffix.lower()
                })
    return records

print("Scanning for datasets...")
img_map = index_all_images(CONFIG["search_roots"])
records = discover_csv_records(CONFIG["search_roots"], img_map)
if not records:
    records = discover_folder_records(CONFIG["search_roots"])

if not records:
    raise RuntimeError("Could not find any annotated dataset (CSV or class folders) under search roots.")

raw_df = pd.DataFrame(records)
print(f"Total discovered images: {len(raw_df)}")
print(raw_df["class"].value_counts().to_string())

# ----------------- Quality & Inspection -----------------
from PIL import Image, ImageFile, UnidentifiedImageError
import scipy.fftpack as fftpack

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = False

def inspect_image(path: str) -> dict:
    info = {"readable": False, "width": None, "height": None, "mode": None, "file_size": None, "error": None}
    try:
        info["file_size"] = os.path.getsize(path)
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            info.update(width=im.width, height=im.height, mode=im.mode)
            im.load()
        info["readable"] = True
    except Exception as ex:
        info["error"] = str(ex)
    return info

print("\nInspecting image decodability...")
insp = [inspect_image(p) for p in raw_df["filepath"]]
insp_df = pd.DataFrame(insp)
index_df = pd.concat([raw_df.reset_index(drop=True), insp_df], axis=1)

bad = index_df[~index_df["readable"]]
CLEAN = index_df[index_df["readable"]].copy().reset_index(drop=True)
CLEAN["aspect_ratio"] = CLEAN["width"] / CLEAN["height"]
CLEAN["file_size_kb"] = CLEAN["file_size"] / 1024.0
print(f"Decodable clean images: {len(CLEAN)} (Unreadable/Corrupt excluded: {len(bad)})")

# ----------------- Deduplication & Hashing -----------------
def sha256_of_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def phash_bits(path: str, hash_size: int = 8, highfreq_factor: int = 4):
    img_size = hash_size * highfreq_factor
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.float64)
        dct = fftpack.dct(fftpack.dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
        low = dct[:hash_size, :hash_size].flatten()
        med = np.median(low[1:])
        return low > med
    except Exception:
        return None

print("Computing file hashes for deduplication...")
sha_list = [sha256_of_file(p) for p in CLEAN["filepath"]]
ph_bits = [phash_bits(p, CONFIG["phash_size"]) for p in CLEAN["filepath"]]
CLEAN["sha256"] = sha_list

parent = list(range(len(CLEAN)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

# Exact duplicates
for sha, idxs in CLEAN.groupby("sha256").groups.items():
    idxs = list(idxs)
    for j in idxs[1:]:
        union(idxs[0], j)

# Near duplicates
if CONFIG["run_near_dup"]:
    valid = [i for i, b in enumerate(ph_bits) if b is not None]
    thr = CONFIG["near_dup_hamming_max"]
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            diff = np.count_nonzero(ph_bits[valid[i]] != ph_bits[valid[j]])
            if diff <= thr:
                union(valid[i], valid[j])

CLEAN["dup_group"] = [find(i) for i in range(len(CLEAN))]
print(f"Unique duplicate-isolated components: {CLEAN['dup_group'].nunique()}")

# ----------------- Stratified Leakage-Free Split -----------------
rng = random.Random(SEED)
comp_rows = []
for gid, sub in CLEAN.groupby("dup_group"):
    maj = sub["class"].value_counts().idxmax()
    comp_rows.append({"dup_group": gid, "class": maj, "size": len(sub)})
comp_df = pd.DataFrame(comp_rows)

ratios = CONFIG["split"]
total_clean = len(CLEAN)
target_val = int(round(total_clean * ratios["val"]))
target_test = int(round(total_clean * ratios["test"]))

comp_split = {}
for cls in CLASSES:
    sub_comps = comp_df[comp_df["class"] == cls].copy()
    items = list(sub_comps.to_dict("records"))
    rng.shuffle(items)
    n_cls = sum(it["size"] for it in items)
    cls_target_val = int(round(n_cls * ratios["val"]))
    cls_target_test = int(round(n_cls * ratios["test"]))
    val_cur, test_cur = 0, 0
    for it in items:
        if val_cur < cls_target_val:
            comp_split[it["dup_group"]] = "val"
            val_cur += it["size"]
        elif test_cur < cls_target_test:
            comp_split[it["dup_group"]] = "test"
            test_cur += it["size"]
        else:
            comp_split[it["dup_group"]] = "train"

CLEAN["split"] = CLEAN["dup_group"].map(comp_split)
train_df = CLEAN[CLEAN["split"] == "train"].reset_index(drop=True)
val_df = CLEAN[CLEAN["split"] == "val"].reset_index(drop=True)
test_df = CLEAN[CLEAN["split"] == "test"].reset_index(drop=True)

print(f"\nSplit complete -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ----------------- Image Transforms & Loader -----------------
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

def to_rgb(img: Image.Image) -> Image.Image:
    if getattr(img, "is_animated", False):
        img.seek(0)
    if img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        return bg
    return img.convert("RGB")

def fit_pixel_budget(img: Image.Image, min_p: int, max_p: int) -> Image.Image:
    w, h = img.size
    p = w * h
    if p > max_p:
        scale = math.sqrt(max_p / p)
        nw, nh = max(28, int(round(w * scale / 28) * 28)), max(28, int(round(h * scale / 28) * 28))
        return img.resize((nw, nh), Image.Resampling.LANCZOS)
    return img

color_jitter = T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)

def load_image(path: str, train: bool = False) -> Image.Image:
    with Image.open(path) as im:
        img = to_rgb(im)
    img = fit_pixel_budget(img, CONFIG["min_pixels"], CONFIG["max_pixels"])
    if train:
        if random.random() < 0.30:
            deg = random.uniform(-5.0, 5.0)
            img = img.rotate(deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(255, 255, 255))
        if random.random() < 0.30:
            img = color_jitter(img)
    return img

# ----------------- Model & Tokenizer -----------------
import transformers
from transformers import AutoProcessor, AutoConfig, BitsAndBytesConfig

print(f"\nLoading Model Processor: {CONFIG['model_id']} ...")
try:
    processor = AutoProcessor.from_pretrained(
        CONFIG["model_id"], trust_remote_code=True,
        min_pixels=CONFIG["min_pixels"], max_pixels=CONFIG["max_pixels"])
except Exception:
    processor = AutoProcessor.from_pretrained(CONFIG["model_id"], trust_remote_code=True)

tokenizer = processor.tokenizer
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

class_list_str = ", ".join(CLASSES)
CLASSIFICATION_PROMPT = (
    f"You are analysing a meme circulated on Bangladeshi social media.\n"
    f"Classify this meme into exactly one of the following categories:\n"
    f"{class_list_str}.\n"
    f"Answer with the category name only."
)

def build_messages(image):
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": CLASSIFICATION_PROMPT}]}]

class MemeDataset(Dataset):
    def __init__(self, df, train=False, with_answer=True):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.with_answer = with_answer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = load_image(r["filepath"], train=self.train)
        msgs = build_messages(img)
        prefix = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if self.with_answer:
            full_text = prefix + r["class"] + tokenizer.eos_token
            proc_p = processor(text=[prefix], images=[img], return_tensors="pt")
            prompt_len = proc_p["input_ids"].shape[1]
            proc_full = processor(text=[full_text], images=[img], return_tensors="pt")
            item = {}
            for k, v in proc_full.items():
                if k in ("input_ids", "attention_mask"):
                    item[k] = v.squeeze(0)
                elif k == "image_grid_thw":
                    item[k] = v if v.dim() == 2 else v.view(-1, 3)
                else:
                    item[k] = v.squeeze(0) if v.dim() > 2 else v
            item["prompt_len"] = prompt_len
        else:
            proc = processor(text=[prefix], images=[img], return_tensors="pt")
            item = {}
            for k, v in proc.items():
                if k in ("input_ids", "attention_mask"):
                    item[k] = v.squeeze(0)
                elif k == "image_grid_thw":
                    item[k] = v if v.dim() == 2 else v.view(-1, 3)
                else:
                    item[k] = v.squeeze(0) if v.dim() > 2 else v
        item["row_index"] = idx
        item["label_id"] = int(r["label_id"])
        return item

VISUAL_KEYS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")

def collate(batch):
    L = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), L), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attn[i, :n] = b["attention_mask"]
        if "prompt_len" in b:
            lab = b["input_ids"].clone()
            lab[: b["prompt_len"]] = -100
            labels[i, :n] = lab
    out = {"input_ids": input_ids, "attention_mask": attn}
    if any("prompt_len" in b for b in batch):
        out["labels"] = labels
    for k in VISUAL_KEYS:
        if k in batch[0]:
            tensors = [b[k] for b in batch]
            if k == "image_grid_thw":
                tensors = [t.view(-1, 3) if t.dim() < 2 else t for t in tensors]
                out[k] = torch.cat(tensors, dim=0)
            elif k == "pixel_values":
                out[k] = torch.cat(tensors, dim=0)
            else:
                out[k] = torch.cat(tensors, dim=0)
    out["row_index"] = torch.tensor([b["row_index"] for b in batch])
    out["label_id"] = torch.tensor([b["label_id"] for b in batch])
    return out

# ----------------- Model Loading & PEFT -----------------
BF16_OK = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
COMPUTE_DTYPE = torch.bfloat16 if BF16_OK else torch.float16
TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0.0

HAS_BNB = False
try:
    import bitsandbytes
    HAS_BNB = True
except Exception:
    pass

USE_4BIT = bool(HAS_BNB and TOTAL_VRAM and TOTAL_VRAM < 15.5)
print(f"GPU VRAM: {TOTAL_VRAM:.1f} GiB | Compute Dtype: {COMPUTE_DTYPE} | BitsAndBytes Available: {HAS_BNB} | Use 4-Bit: {USE_4BIT}")

def load_base_model():
    kwargs = dict(dtype=COMPUTE_DTYPE, device_map="auto" if torch.cuda.is_available() else None,
                  trust_remote_code=True, attn_implementation=CONFIG["attn_implementation"])
    if USE_4BIT and HAS_BNB and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE, bnb_4bit_use_double_quant=True)
    loaders = [
        ("Qwen2_5_VLForConditionalGeneration", lambda: getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)),
        ("Qwen2VLForConditionalGeneration", lambda: getattr(transformers, "Qwen2VLForConditionalGeneration", None)),
        ("Qwen3VLForConditionalGeneration", lambda: getattr(transformers, "Qwen3VLForConditionalGeneration", None)),
        ("AutoModelForImageTextToText", lambda: getattr(transformers, "AutoModelForImageTextToText", None)),
        ("AutoModelForVision2Seq", lambda: getattr(transformers, "AutoModelForVision2Seq", None)),
        ("AutoModelForCausalLM", lambda: getattr(transformers, "AutoModelForCausalLM", None)),
    ]
    for name, get_cls in loaders:
        cls = get_cls()
        if cls is not None:
            try:
                print(f"Loading via {name}...")
                return cls.from_pretrained(CONFIG["model_id"], **kwargs)
            except Exception as e:
                print(f"  {name} failed: {e}")
    raise RuntimeError(f"Could not load {CONFIG['model_id']}.")

base_model = load_base_model()
DEVICE = next(base_model.parameters()).device

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

if USE_4BIT and torch.cuda.is_available():
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=CONFIG["train"]["gradient_checkpointing"])

lora_cfg = LoraConfig(
    r=CONFIG["lora"]["r"],
    lora_alpha=CONFIG["lora"]["alpha"],
    lora_dropout=CONFIG["lora"]["dropout"],
    bias=CONFIG["lora"]["bias"],
    target_modules=CONFIG["lora"]["target_modules"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_cfg)
if CONFIG["train"]["gradient_checkpointing"] and torch.cuda.is_available():
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.print_trainable_parameters()

# ----------------- Scoring & Metrics -----------------
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABEL_TOKEN_IDS = {c: tokenizer(c, add_special_tokens=False)["input_ids"] for c in CLASSES}
FIRST_TOKENS = [ids[0] for ids in LABEL_TOKEN_IDS.values()]
FAST_SCORING = len(set(FIRST_TOKENS)) == NUM_CLASSES
FIRST_TOKEN_TENSOR = torch.tensor(FIRST_TOKENS, dtype=torch.long, device=DEVICE)

@torch.no_grad()
def score_dataframe(df, return_loss=False):
    model.eval()
    dl = DataLoader(MemeDataset(df, train=False, with_answer=False),
                    batch_size=CONFIG["train"]["eval_batch_images"], shuffle=False,
                    num_workers=2 if torch.cuda.is_available() else 0, collate_fn=collate)
    all_probs = []
    tot_loss, n_samples = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    
    for batch in dl:
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k not in ("row_index", "label_id")}
        with torch.autocast("cuda", dtype=COMPUTE_DTYPE, enabled=torch.cuda.is_available()):
            out = model(**inputs, output_hidden_states=False, use_cache=False)
        
        attn_mask = inputs["attention_mask"]
        seq_lens = (attn_mask.sum(dim=1) - 1).clamp(min=0)
        batch_idx = torch.arange(len(seq_lens), device=DEVICE)
        last_logits = out.logits[batch_idx, seq_lens, :].float()
        
        scores = last_logits[:, FIRST_TOKEN_TENSOR]
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=1.0 / NUM_CLASSES, posinf=1.0, neginf=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        all_probs.append(probs.cpu())
        
        if return_loss and "label_id" in batch:
            target_ids = batch["label_id"].to(DEVICE)
            loss_val = loss_fn(scores, target_ids)
            tot_loss += float(loss_val.item())
            n_samples += len(target_ids)
            
        del out, inputs
        
    probs = torch.cat(all_probs, dim=0).numpy()
    probs = np.nan_to_num(probs, nan=1.0 / NUM_CLASSES, posinf=1.0, neginf=0.0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-9, None)
    preds = probs.argmax(axis=1)
    
    if return_loss:
        mean_val_loss = tot_loss / max(n_samples, 1)
        return probs, preds, mean_val_loss
    return probs, preds

# ----------------- Training Loop -----------------
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import copy

TR = CONFIG["train"]
train_loader = DataLoader(MemeDataset(train_df, train=True, with_answer=True),
                          batch_size=TR["per_device_batch_size"], shuffle=True,
                          num_workers=2 if torch.cuda.is_available() else 0,
                          collate_fn=collate, pin_memory=torch.cuda.is_available())

steps_per_epoch = math.ceil(len(train_loader) / TR["grad_accum_steps"])
total_steps = steps_per_epoch * TR["epochs"]
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=TR["lr"], weight_decay=TR["weight_decay"])
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * TR["warmup_ratio"]), num_training_steps=total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=(COMPUTE_DTYPE == torch.float16 and torch.cuda.is_available()))

best = {"macro_f1": -1.0, "epoch": -1}
print(f"\nStarting Training: {TR['epochs']} epochs, {total_steps} optimizer steps...")

for epoch in range(1, TR["epochs"] + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss, seen = 0.0, 0
    t_ep = time.time()
    for it, batch in enumerate(train_loader):
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k not in ("row_index", "label_id")}
        with torch.autocast("cuda", dtype=COMPUTE_DTYPE, enabled=torch.cuda.is_available()):
            out = model(**inputs, use_cache=False)
            loss = out.loss / TR["grad_accum_steps"]
        scaler.scale(loss).backward()
        running_loss += float(out.loss)
        seen += 1

        if (it + 1) % TR["grad_accum_steps"] == 0 or (it + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], TR["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    # Validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    val_probs, val_preds = score_dataframe(val_df)
    val_acc = accuracy_score(val_df["label_id"], val_preds)
    val_f1 = f1_score(val_df["label_id"], val_preds, average="macro", zero_division=0)
    print(f"[Epoch {epoch}/{TR['epochs']}] Time: {(time.time()-t_ep)/60:.1f}m | Loss: {running_loss/max(seen,1):.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

    if val_f1 > best["macro_f1"]:
        best = {"macro_f1": val_f1, "epoch": epoch}
        model.save_pretrained(OUT / "model" / "best_adapter")
        print(f"  -> Saved new best checkpoint (Val Macro-F1: {best['macro_f1']:.4f})")

# ----------------- Final Test Evaluation -----------------
print("\n" + "=" * 60)
print(f"Evaluating Best Model on Held-out Test Set ({len(test_df)} samples)...")
print("=" * 60)

test_probs, test_preds = score_dataframe(test_df)
y_true = test_df["label_id"].values
y_pred = test_preds

test_acc = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
test_wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"\nFinal Test Accuracy   : {test_acc:.4f}")
print(f"Final Test Macro-F1   : {test_f1:.4f}")
print(f"Final Test Weighted-F1: {test_wf1:.4f}\n")

report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
print(report)

# Save predictions and metrics
test_df["predicted_class"] = [ID_TO_CLASS[p] for p in test_preds]
for ci, cls in enumerate(CLASSES):
    test_df[f"prob_{cls}"] = test_probs[:, ci]
test_df.to_csv(OUT / "predictions" / "test_predictions.csv", index=False)

with open(OUT / "metrics" / "test_metrics.json", "w") as f:
    json.dump({
        "accuracy": float(test_acc),
        "macro_f1": float(test_f1),
        "weighted_f1": float(test_wf1),
        "best_epoch": int(best["epoch"]),
        "best_val_macro_f1": float(best["macro_f1"]),
    }, f, indent=2)

print(f"\nAll artifacts saved successfully to {OUT.resolve()}")
