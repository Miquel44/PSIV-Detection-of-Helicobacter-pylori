#!/usr/bin/env python3
# train_phase3_patient_diagnosis.py
# Diagnóstico por paciente usando MIL con atención sobre patches.

import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from AutoEncoder.Models.models import OneShotCNNNet
from Attention.AttentionUnits import Attention, GatedAttention

# ================================
# CONFIGURACIÓN
# ================================
# CSV con columnas: CODI,DENSITAT (NEGATIVA | BAIXA | ALTA)
PATIENT_DIAGNOSIS_CSV = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Datasets/PatientDiagnosis.csv"

MODE = "emb_only"  # "emb_only" | "rep_attention" | "rep_with_cls"

CROPPED_ROOT = "/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/"
PRECOMPUTED_NPZ = "./precomputed_patient_embeddings.npz"

BACKBONE_CKPT = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/backbone_phase1.pth"
PATCH_CLASSIFIER_CKPT = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/patch_classifier_phase2.pth"

OUTPUT_PATIENT_MODEL = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/patient_model_phase3.pth"

IMAGE_SIZE = 256
MAX_PATCHES_PER_PATIENT = None
BATCH_SIZE_BAGS = 1
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
PATIENCE = 5
SEED = 42

ATTENTION_TYPE = "gated"  # "simple" | "gated"
DECOM_SPACE = 128
ATTENTION_BRANCHES = 1
CLF_HIDDEN = 128
DROP = 0.5

# ================================
# Utils
# ================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_pos_weight(labels):
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0:
        return 1.0
    total = pos + neg
    return float((total / (2.0 * max(pos, 1))) * 0.5)

def load_patient_label_map(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    mapping = {"NEGATIVA": 0, "BAIXA": 1, "ALTA": 1}
    label_map = {}
    for _, row in df.iterrows():
        code = str(row["CODI"]).strip()
        dens = str(row["DENSITAT"]).strip().upper()
        if dens in mapping:
            label_map[code] = mapping[dens]
    return label_map

# ================================
# Datasets
# ================================
class PatientBagsFromNPZ(Dataset):
    def __init__(self, npz_path, label_map, max_patches=None):
        super().__init__()
        self.data = np.load(npz_path, allow_pickle=True)
        self.max_patches = max_patches
        grouped = {}
        for k in self.data.files:
            code = k.split("_")[0]
            grouped.setdefault(code, []).append(k)
        self.items = []
        for code, lbl in label_map.items():
            if code in grouped:
                mats = []
                for key in grouped[code]:
                    arr = self.data[key]
                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 0:
                        mats.append(arr)
                if len(mats) == 0:
                    continue
                cat = np.concatenate(mats, axis=0)
                self.items.append((code, lbl, cat))
        self.items = [it for it in self.items if it[2].shape[0] > 0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        code, label, emb = self.items[idx]
        if self.max_patches is not None and emb.shape[0] > self.max_patches:
            sel = np.random.choice(emb.shape[0], self.max_patches, replace=False)
            emb = emb[sel]
        emb = torch.tensor(emb, dtype=torch.float32)
        return emb, torch.tensor(label, dtype=torch.float32), code

class PatientBagsFromImages(Dataset):
    def __init__(self, cropped_root, label_map, image_size=256, max_patches=None):
        super().__init__()
        self.root = cropped_root
        self.tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self.max_patches = max_patches
        folders = [f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))]
        prefix_map = {}
        for f in folders:
            code = f.split("_")[0]
            prefix_map.setdefault(code, []).append(f)
        self.items = []
        for code, lbl in label_map.items():
            if code not in prefix_map:
                continue
            paths = []
            for folder in prefix_map[code]:
                d = os.path.join(self.root, folder)
                imgs = [os.path.join(d, x) for x in os.listdir(d) if x.endswith(".png")]
                paths.extend(imgs)
            if len(paths) == 0:
                continue
            self.items.append((code, lbl, paths))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        code, label, paths = self.items[idx]
        if self.max_patches is not None and len(paths) > self.max_patches:
            paths = list(np.random.choice(paths, self.max_patches, replace=False))
        imgs = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(self.tf(img))
            except Exception:
                continue
        if len(imgs) == 0:
            x = torch.zeros(0, 3, IMAGE_SIZE, IMAGE_SIZE)
        else:
            x = torch.stack(imgs, dim=0)
        return x, torch.tensor(label, dtype=torch.float32), code

def bag_collate(batch):
    return batch[0]

# ================================
# Modelos
# ================================
class MILAttentionClassifier(nn.Module):
    def __init__(self, in_features, decom_space=128, attention_branches=1,
                 attention_type="gated", clf_hidden=128, drop=0.5):
        super().__init__()
        net_params = {
            'in_features': in_features,
            'decom_space': decom_space,
            'ATTENTION_BRANCHES': attention_branches
        }
        self.att = GatedAttention(net_params) if attention_type == "gated" else Attention(net_params)
        clf_in = attention_branches * in_features
        self.clf = nn.Sequential(
            nn.Linear(clf_in, clf_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(clf_hidden, 1)
        )

    def forward(self, H):
        if H.ndim != 2 or H.shape[0] == 0:
            return torch.tensor(-10.0, device=H.device), None, None
        Z, A = self.att(H)
        logit = self.clf(Z.reshape(-1))
        return logit.squeeze(0), Z, A

class PatchClassifier(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.backbone = OneShotCNNNet(
            {'num_input_channels': 3},
            {'dim': 2, 'drop_rate': 0.3, 'block_configs': [[32, 32], [64, 64], [128, 128]]},
            {'prct': [1], 'hidden': [128], 'dropout': 0.5}
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def feature(self, x):
        return self.backbone.feature_extraction(x)

    def forward(self, x):
        with torch.no_grad():
            emb = self.backbone.feature_extraction(x)
        return self.classifier(emb).squeeze(1)

# ================================
# Evaluación
# ================================
def evaluate(model, loader, device, encode_fn):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, lab, pid in loader:
            lab = lab.to(device)
            if isinstance(data, torch.Tensor) and data.ndim == 2:
                feats = data.to(device)
            else:
                feats = encode_fn(data.to(device))
            logit, _, _ = model(feats)
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0
            y_true.append(int(lab.item()))
            y_pred.append(pred)
    report = classification_report(y_true, y_pred, target_names=['Negative','Positive'], digits=4)
    cm = confusion_matrix(y_true, y_pred)
    acc = (np.array(y_true) == np.array(y_pred)).mean()
    return acc, report, cm

# ================================
# Main
# ================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    assert MODE in ("emb_only", "rep_attention", "rep_with_cls")

    label_map = load_patient_label_map(PATIENT_DIAGNOSIS_CSV)
    print(f"Pacientes etiquetados: {len(label_map)}")

    if MODE == "emb_only":
        assert os.path.exists(PRECOMPUTED_NPZ), f"No existe NPZ: {PRECOMPUTED_NPZ}"
        full_ds = PatientBagsFromNPZ(PRECOMPUTED_NPZ, label_map, MAX_PATCHES_PER_PATIENT)
        assert len(full_ds) > 0, "NPZ sin pacientes válidos."
        sample_feats, sample_lab, _ = full_ds[0]
        emb_dim = int(sample_feats.shape[1])
        print(f"Dim embedding (NPZ): {emb_dim}")
        idxs = np.arange(len(full_ds))
        np.random.shuffle(idxs)
        n_val = int(math.ceil(VAL_SPLIT * len(full_ds)))
        val_idx = set(idxs[:n_val])
        train_items, val_items = [], []
        for i in range(len(full_ds)):
            (x, y, pid) = full_ds[i]
            (val_items if i in val_idx else train_items).append((x, y, pid))

        class WrapDS(Dataset):
            def __init__(self, items): self.items = items
            def __len__(self): return len(self.items)
            def __getitem__(self, i): return self.items[i]

        train_ds = WrapDS(train_items)
        val_ds = WrapDS(val_items)

        def encode_fn(feat_tensor):
            return feat_tensor

    else:
        full_ds = PatientBagsFromImages(CROPPED_ROOT, label_map, IMAGE_SIZE, MAX_PATCHES_PER_PATIENT)
        assert len(full_ds) > 0, "Dataset vacío tras mapear códigos."
        backbone = OneShotCNNNet(
            {'num_input_channels': 3},
            {'dim': 2, 'drop_rate': 0.3, 'block_configs': [[32, 32], [64, 64], [128, 128]]},
            {'prct': [1], 'hidden': [128], 'dropout': 0.5}
        )
        backbone.load_state_dict(torch.load(BACKBONE_CKPT, map_location="cpu"))
        backbone.to(device)
        backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
            emb_dim = backbone.feature_extraction(dummy).shape[1]
        print(f"Dim embedding (backbone): {emb_dim}")

        if MODE == "rep_with_cls":
            patch_model = PatchClassifier(emb_dim=emb_dim)
            state = torch.load(PATCH_CLASSIFIER_CKPT, map_location="cpu")
            patch_model.load_state_dict(state, strict=False)
            patch_model.to(device)
            patch_model.eval()

            def encode_fn(batch_imgs):
                if batch_imgs.shape[0] == 0:
                    return torch.zeros(0, emb_dim + 1, device=device)
                with torch.no_grad():
                    emb = backbone.feature_extraction(batch_imgs)
                    logits = patch_model.classifier(emb).squeeze(1)
                    feats = torch.cat([emb, logits.unsqueeze(1)], dim=1)
                return feats

            emb_dim = emb_dim + 1
        else:
            def encode_fn(batch_imgs):
                if batch_imgs.shape[0] == 0:
                    return torch.zeros(0, emb_dim, device=device)
                with torch.no_grad():
                    return backbone.feature_extraction(batch_imgs)

        idxs = np.arange(len(full_ds))
        np.random.shuffle(idxs)
        n_val = int(math.ceil(VAL_SPLIT * len(full_ds)))
        val_idx = set(idxs[:n_val])

        class IndexDS(Dataset):
            def __init__(self, ds, idc): self.ds, self.idc = ds, list(idc)
            def __len__(self): return len(self.idc)
            def __getitem__(self, i): return self.ds[self.idc[i]]

        train_ds = IndexDS(full_ds, [i for i in range(len(full_ds)) if i not in val_idx])
        val_ds = IndexDS(full_ds, [i for i in range(len(full_ds)) if i in val_idx])

    print(f"Train patients: {len(train_ds)} | Val patients: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_BAGS, shuffle=True, num_workers=4, collate_fn=bag_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_BAGS, shuffle=False, num_workers=4, collate_fn=bag_collate)

    mil_model = MILAttentionClassifier(
        in_features=emb_dim,
        decom_space=DECOM_SPACE,
        attention_branches=ATTENTION_BRANCHES,
        attention_type=ATTENTION_TYPE,
        clf_hidden=CLF_HIDDEN,
        drop=DROP
    ).to(device)

    all_labels = []
    for _, lab, _ in train_ds:
        all_labels.append(int(lab))
    pos_weight = compute_pos_weight(np.array(all_labels))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.Adam(mil_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc = 0.0
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        mil_model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for data, lab, pid in pbar:
            lab = lab.to(device)
            if isinstance(data, torch.Tensor) and data.ndim == 2:
                feats = data.to(device)
            else:
                feats = encode_fn(data.to(device))
            optimizer.zero_grad()
            logit, _, _ = mil_model(feats)
            loss = criterion(logit.view(1), lab.view(1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        avg_loss = float(np.mean(losses)) if losses else 0.0
        acc, report, cm = evaluate(mil_model, val_loader, device, encode_fn)
        print(f"\nEpoch {epoch} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        print(report)
        print("Confusion Matrix:\n", cm)

        if acc > best_acc:
            best_acc = acc
            patience_ctr = 0
            os.makedirs(os.path.dirname(OUTPUT_PATIENT_MODEL), exist_ok=True)
            torch.save({
                "state_dict": mil_model.state_dict(),
                "config": {
                    "MODE": MODE,
                    "ATTENTION_TYPE": ATTENTION_TYPE,
                    "DECOM_SPACE": DECOM_SPACE,
                    "ATTENTION_BRANCHES": ATTENTION_BRANCHES,
                    "CLF_HIDDEN": CLF_HIDDEN,
                    "DROP": DROP,
                    "EMB_DIM": emb_dim,
                    "IMAGE_SIZE": IMAGE_SIZE
                }
            }, OUTPUT_PATIENT_MODEL)
            print(f"Saved best model: {OUTPUT_PATIENT_MODEL}")
        else:
            patience_ctr += 1
            print(f"No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print("Early stopping.")
                break

    if os.path.exists(OUTPUT_PATIENT_MODEL):
        ckpt = torch.load(OUTPUT_PATIENT_MODEL, map_location=device)
        mil_model.load_state_dict(ckpt["state_dict"])
    acc, report, cm = evaluate(mil_model, val_loader, device, encode_fn)
    print("\nFINAL EVALUATION")
    print(f"Val Acc: {acc:.4f}")
    print(report)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()