# requirements: pip install rasterio numpy pandas torch matplotlib tqdm
import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

# ── 1. CONSTANTS ────────────────────────────────────────────────────────────────
# Sentinel-2 bands stored in MARIDA .tif files (in order, 0-indexed)
# B1=Coastal, B2=Blue, B3=Green, B4=Red, B5-B7=RedEdge, B8=NIR,
# B8A=NarrowNIR, B11=SWIR1, B12=SWIR2
BAND_NAMES = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
N_BANDS    = 11

# MARIDA class labels (from labels_mapping.txt)
CLASS_LABELS = {
    0: 'Unknown',
    1: 'Marine Debris',
    2: 'Dense Sargassum',
    3: 'Sparse Sargassum',
    4: 'Natural Organic Material',
    5: 'Ship',
    6: 'Clouds',
    7: 'Marine Water',
    8: 'Sediment-Laden Water',
    9: 'Foam',
    10: 'Turbid Water',
    11: 'Shallow Water',
    12: 'Waves',
    13: 'Cloud Shadows',
    14: 'Wakes',
    15: 'Mixed Water',
}
# treat class 1 (Marine Debris — contains microplastics) as anomaly=1, rest=0

DATASET_ROOT = Path("marida_dataset")
PATCHES_DIR  = DATASET_ROOT / "patches"
SPLITS_DIR   = DATASET_ROOT / "splits"

# ── 2. LOAD SPLIT LISTS ──────────────────────────────────────────────────────────
def load_split(split: str) -> list[str]:
    """
    Reads train_X.txt / val_X.txt / test_X.txt.
    Each line is a patch ID like: S2_9-10-17_16PEC_9
    Returns list of patch IDs.
    """
    path = SPLITS_DIR / f"{split}_X.txt"
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

train_ids = load_split("train")
val_ids   = load_split("val")
test_ids  = load_split("test")
print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

# ── 3. PATCH RESOLVER ───────────────────────────────────────────────────────────
def resolve_patch_paths(patch_id: str) -> dict:
    """
    split file:  1-12-19_48MYU_0
    folder:      S2_1-12-19_48MYU
    files:       S2_1-12-19_48MYU_0.tif, S2_1-12-19_48MYU_0_cl.tif, ...
    """
    parts         = patch_id.rsplit('_', 1)   # ['1-12-19_48MYU', '0']
    scene_id      = parts[0]                   # '1-12-19_48MYU'
    scene_dir     = PATCHES_DIR / f"S2_{scene_id}"   # patches/S2_1-12-19_48MYU
    full_patch_id = f"S2_{patch_id}"           # 'S2_1-12-19_48MYU_0'

    return {
        "bands":  scene_dir / f"{full_patch_id}.tif",
        "labels": scene_dir / f"{full_patch_id}_cl.tif",
        "conf":   scene_dir / f"{full_patch_id}_conf.tif",
    }
# ── 4. SINGLE PATCH READER ──────────────────────────────────────────────────────
def read_patch(patch_id: str) -> dict:
    """
    Returns:
        bands  : np.float32 (H, W, 11)  — raw DN values (divide by 10000 → reflectance)
        labels : np.int8    (H, W)       — class index per pixel
        conf   : np.int8    (H, W)       — 1=high, 2=med, 3=low confidence
    """
    paths = resolve_patch_paths(patch_id)

    with rasterio.open(paths["bands"]) as src:
        bands = src.read(masked=False).astype(np.float32)  # masked=False prevents nodata→0
        bands = np.transpose(bands, (1, 2, 0))  # → (H, W, 11)

    with rasterio.open(paths["labels"]) as src:
        labels = src.read(1).astype(np.int16)    # (H, W)

    with rasterio.open(paths["conf"]) as src:
        conf = src.read(1).astype(np.int16)      # (H, W)

    return {"bands": bands, "labels": labels, "conf": conf, "id": patch_id}

# ── 5. QUICK INSPECTION ─────────────────────────────────────────────────────────
sample = read_patch(train_ids[0])

print(f"Bands shape  : {sample['bands'].shape}")   # e.g. (256, 256, 11)
print(f"Labels shape : {sample['labels'].shape}")
print(f"Reflectance range: [{sample['bands'].min():.4f}, {sample['bands'].max():.4f}]")
print(f"Unique classes in patch: {np.unique(sample['labels'])}")

print(f"Raw band dtype before /10000: check .tif directly")
# Check raw uint16 values BEFORE dividing
with rasterio.open(resolve_patch_paths(train_ids[0])["bands"]) as src:
    raw = src.read()
    print(f"Raw dtype: {raw.dtype}")
    print(f"Raw value range: [{raw.min()}, {raw.max()}]")
    print(f"Nodata value: {src.nodata}")

# ── 6. SPECTRAL INDICES (the real signal) ───────────────────────────────────────
def compute_spectral_indices(bands: np.ndarray) -> np.ndarray:
    """
    bands: (H, W, 11) reflectance
    Appends 3 physics-derived indices as extra channels → (H, W, 14)

    FAI  (Floating Algae Index): isolates floating material from water background
    PI   (Plastic Index): B8A/(B8A+B4), tuned to plastic spectral response
    NDVI (sanity check): vegetation vs water separation
    """
    eps = 1e-6  # larger epsilon as 1e-8 was too small vs float32 precision
    B4   = bands[..., 3]   # Red      ~665nm
    B6   = bands[..., 5]   # RedEdge2 ~740nm  (FAI baseline interpolation)
    B8   = bands[..., 7]   # NIR      ~842nm
    B8A  = bands[..., 8]   # NIR2     ~865nm
    B11  = bands[..., 9]   # SWIR1    ~1610nm
    B12  = bands[..., 10]  # SWIR2    ~2190nm

    # FAI = B8 - [B4 + (B11-B4) * (λ8-λ4)/(λ11-λ4)]
    # wavelength-interpolated baseline between Red and SWIR1
    FAI  = B8 - (B4 + (B11 - B4) * (842 - 665) / (1610 - 665))

    # PI = B8A / (B8A + B4)  — plastic particles reflect strongly in NIR, absorb in Red
    PI   = B8A / (B8A + B4 + eps)

    # NDVI = (B8 - B4) / (B8 + B4)
    NDVI = (B8 - B4) / (B8 + B4 + eps)

    extra = np.stack([FAI, PI, NDVI], axis=-1)          # (H, W, 3)
    return np.concatenate([bands, extra], axis=-1)       # (H, W, 14)

    # Kill any remaining NaN/Inf — replace with 0 (numerically neutral), this is not doing shit
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

# ── 7. PYTORCH DATASET ──────────────────────────────────────────────────────────
class MARIDASpectralDataset(Dataset):
    """
    Flattens each patch into a bag of pixel-level spectral vectors.
    Each item = one pixel's spectral signature + binary anomaly label.

    For the VAE we train ONLY on clean water pixels (label != 1),
    so the model learns the normal manifold.
    Anomaly detection = high reconstruction error at test time.
    """
    def __init__(
        self,
        patch_ids: list[str],
        anomaly_class: int = 1,           # Marine Debris
        conf_threshold: int = 2,          # only use pixels with conf ≤ threshold
        train_mode: bool = True,          # True → exclude anomalies from data
        use_indices: bool = True,
    ):
        self.vectors = []   # spectral feature vectors
        self.targets = []   # 0=normal, 1=anomaly

        for pid in patch_ids:
            patch = read_patch(pid)
            feat  = compute_spectral_indices(patch["bands"]) if use_indices \
                    else patch["bands"]                            # (H, W, C)
            H, W, C = feat.shape
            flat_feat = feat.reshape(-1, C)                       # (H*W, C)
            flat_lbl  = patch["labels"].reshape(-1)               # (H*W,)
            flat_conf = patch["conf"].reshape(-1)                 # (H*W,)

            # conf=255 is nodata sentinel in some rasters — exclude explicitly
            conf_mask = (flat_conf >= 1) & (flat_conf <= conf_threshold)

            binary_lbl = (flat_lbl == anomaly_class).astype(np.int8)

            if train_mode:
                # train only on high-confidence NORMAL pixels
                mask = conf_mask & (binary_lbl == 0)
            else:
                # val/test: use all high-confidence pixels
                mask = conf_mask
            if mask.sum() == 0:
                continue  # skip patches with no valid pixels after filtering    
            
            self.vectors.append(flat_feat[mask])
            self.targets.append(binary_lbl[mask])

        self.vectors = np.concatenate(self.vectors, axis=0).astype(np.float32)
        self.targets = np.concatenate(self.targets, axis=0).astype(np.int8)

        n_anomaly = self.targets.sum()
        n_total   = len(self.targets)
        print(f"  Loaded {n_total:,} pixels | "
              f"anomaly: {n_anomaly:,} ({n_anomaly/n_total*100:.3f}%)")

    def __len__(self):  return len(self.vectors)
    def __getitem__(self, idx):
        return torch.tensor(self.vectors[idx]), torch.tensor(self.targets[idx])

# ── 8. BUILD DATALOADERS ─────────────────────────────────────────────────────────
print("Building datasets...")
train_ds = MARIDASpectralDataset(train_ids, train_mode=True)
val_ds   = MARIDASpectralDataset(val_ids,   train_mode=False)
test_ds  = MARIDASpectralDataset(test_ids,  train_mode=False)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=2048, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=2048, shuffle=False, num_workers=4)

# Check raw label distribution across ALL train patches
from collections import Counter
label_counts = Counter()
for pid in train_ids[:20]:  # sample first 20 patches
    patch = read_patch(pid)
    vals, cnts = np.unique(patch["labels"], return_counts=True)
    for v, c in zip(vals, cnts):
        label_counts[int(v)] += int(c)

print("Label distribution (first 20 train patches):")
for cls_id, count in sorted(label_counts.items()):
    print(f"  Class {cls_id:2d} ({CLASS_LABELS.get(cls_id,'?'):30s}): {count:,}")

# ── 9. NORMALIZATION STATS (fit on train only) ───────────────────────────────────
mean = torch.tensor(train_ds.vectors.mean(axis=0))
std  = torch.tensor(train_ds.vectors.std(axis=0) + 1e-8)
print(f"Feature dim: {mean.shape[0]} | mean range: [{mean.min():.3f}, {mean.max():.3f}]")
# will use them to normalize inputs before feeding the VAE