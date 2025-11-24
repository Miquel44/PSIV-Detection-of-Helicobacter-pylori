#!/usr/bin/env python3
# train_phase2_patch_classifier.py

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from Triplets.datasets import Standard_Dataset
from AutoEncoder.Models.models import OneShotCNNNet


# ------------------------------------------------------------
#  CONFIGURACIÓN
# ------------------------------------------------------------
ANNOTATED_ROOT = "/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/"
PATCH_EXCEL = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Datasets/HP_WSI-CoordAllAnnotatedPatches.xlsx"
BACKBONE_CKPT = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/backbone_phase1.pth"
OUTPUT_CLASSIFIER = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/patch_classifier_phase2.pth"

IMAGE_SIZE = 256  # ← DEBE coincidir con Phase 1
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
VAL_SPLIT = 0.2
PATIENCE = 5
# ------------------------------------------------------------


def build_annotated_arrays(root_annotated_dir, patch_excel_path, image_size=256):
    """Carga patches anotados con padding correcto"""
    df = pd.read_excel(patch_excel_path)
    df.columns = df.columns.str.strip()

    folder_map = {}
    for f in os.listdir(root_annotated_dir):
        if os.path.isdir(os.path.join(root_annotated_dir, f)):
            key = f.split("_")[0]
            folder_map.setdefault(key, []).append(f)

    images = []
    labels = []
    pos_count = 0
    neg_count = 0

    for idx, row in df.iterrows():
        pres = row.get("Presence", None)
        if pres not in (1, -1):
            continue

        pat_id = str(row.get("Pat_ID")).strip()
        section_id = str(row.get("Section_ID")).strip()
        window = str(row.get("Window_ID")).strip()
        window_padded = window.zfill(5)

        if pat_id not in folder_map:
            continue

        found = False
        for folder in folder_map[pat_id]:
            folder_section = folder.split("_")[-1]
            if folder_section != section_id:
                continue
                
            folder_path = os.path.join(root_annotated_dir, folder)
            fp = os.path.join(folder_path, f"{window_padded}.png")
            
            if not os.path.exists(fp):
                for cand in os.listdir(folder_path):
                    if cand.startswith(window_padded) and cand.endswith(".png"):
                        fp = os.path.join(folder_path, cand)
                        break
                else:
                    continue

            try:
                img = Image.open(fp).convert("RGB").resize((image_size, image_size))
                arr = np.array(img).astype(np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))
                images.append(arr)
                labels.append(1 if pres == 1 else 0)

                if pres == 1: pos_count += 1
                else: neg_count += 1
                    
                found = True
            except Exception as e:
                continue

            break

    print(f"\n📈 Patches cargados:")
    print(f"   ✅ Positivos: {pos_count}")
    print(f"   ✅ Negativos: {neg_count}")
    print(f"   📊 Ratio: {pos_count/(neg_count+1e-8):.2f}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


class PatchClassifier(nn.Module):
    """Clasificador binario sobre embeddings congelados"""
    def __init__(self, backbone, emb_dim=128):
        super().__init__()
        self.backbone = backbone
        # Congelar backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Clasificador simple
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        with torch.no_grad():
            emb = self.backbone.feature_extraction(x)
        logits = self.classifier(emb)
        return logits.squeeze(1)


def compute_class_weights(y_train):
    """Calcular pesos para weighted BCE loss"""
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    total = len(y_train)
    
    # Weight más conservador para evitar colapso
    pos_weight = (total / (2 * pos_count)) * 0.5  # ← Reducido a la mitad
    
    return pos_weight


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️  Dispositivo:", device)

    # 1. Cargar datos
    print("\n📂 Cargando patches anotados...")
    X, y = build_annotated_arrays(ANNOTATED_ROOT, PATCH_EXCEL, IMAGE_SIZE)
    
    # 2. Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=42
    )
    print(f"\n📊 Split:")
    print(f"   Train: {len(y_train)} samples")
    print(f"   Val: {len(y_val)} samples")
    
    # 3. Crear datasets
    train_dataset = Standard_Dataset(X_train, y_train)
    val_dataset = Standard_Dataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 4. Cargar backbone pre-entrenado
    print(f"\n🔧 Cargando backbone desde {BACKBONE_CKPT}")
    backbone = OneShotCNNNet(
        {'num_input_channels': 3},
        {'dim': 2, 'drop_rate': 0.3, 'block_configs': [[32, 32], [64, 64], [128, 128]]},
        {'prct': [1], 'hidden': [128], 'dropout': 0.5}
    )
    backbone.load_state_dict(torch.load(BACKBONE_CKPT, map_location=device))
    backbone.to(device)
    backbone.eval()
    
    # Detectar dimensión real del embedding
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        dummy_emb = backbone.feature_extraction(dummy_input)
        emb_dim = dummy_emb.shape[1]
        print(f"✅ Dimensión real del embedding: {emb_dim}")
    
    # 5. Crear clasificador
    model = PatchClassifier(backbone, emb_dim=emb_dim).to(device)
    
    # 6. Loss con class weights
    pos_weight = compute_class_weights(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    
    # 7. Entrenamiento
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_losses = []
        
        for imgs, labs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labs = imgs.to(device), labs.float().to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).cpu().numpy()
                
                val_preds.extend(preds)
                val_trues.extend(labs.numpy())
        
        val_preds = np.array(val_preds)
        val_trues = np.array(val_trues)
        val_acc = np.mean(val_preds == val_trues)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_CLASSIFIER)
            print(f"  ✅ Mejor modelo guardado (acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Sin mejora ({patience_counter}/{PATIENCE})")
            
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping en epoch {epoch+1}")
            break
    
    # 8. Reporte final - Cargar mejor modelo
    print("\n" + "="*50)
    print("📊 EVALUACIÓN FINAL (Mejor Modelo)")
    print("="*50)
    
    # Cargar el mejor modelo guardado
    model.load_state_dict(torch.load(OUTPUT_CLASSIFIER))
    model.eval()
    
    val_preds = []
    val_trues = []
    
    with torch.no_grad():
        for imgs, labs in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy()
            
            val_preds.extend(preds)
            val_trues.extend(labs.numpy())
    
    val_preds = np.array(val_preds)
    val_trues = np.array(val_trues)
    
    print(classification_report(val_trues, val_preds, target_names=['Negative', 'Positive']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_trues, val_preds))
    print(f"\n✅ Clasificador guardado en: {OUTPUT_CLASSIFIER}")


if __name__ == "__main__":
    main()