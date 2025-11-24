#!/usr/bin/env python3
# train_phase1_backbone.py

import os
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from Triplets.datasets import TripletDataset
from Triplets.triplet_loss import TripletLoss
from AutoEncoder.Models.models import OneShotCNNNet


# ------------------------------------------------------------
#  CONFIGURACIÓN SIN FLAGS: MODIFICA AQUÍ LAS RUTAS Y PARAMS
# ------------------------------------------------------------
ANNOTATED_ROOT = "/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated/"
PATCH_EXCEL = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Datasets/HP_WSI-CoordAllAnnotatedPatches.xlsx"
OUTPUT_BACKBONE = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/backbone_phase1.pth"

IMAGE_SIZE = 256
MAX_SAMPLES_PER_CLASS = None

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
# ------------------------------------------------------------


def build_annotated_arrays(root_annotated_dir, patch_excel_path, max_samples_per_class=None, image_size=256):
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
        
        # 🔥 FIX: Formatear Window_ID con padding de 5 dígitos
        window_padded = window.zfill(5)  # "902" → "00902"

        if pat_id not in folder_map:
            continue

        found = False
        for folder in folder_map[pat_id]:
            # Verificar que la sección coincida
            folder_section = folder.split("_")[-1]
            if folder_section != section_id:
                continue
                
            folder_path = os.path.join(root_annotated_dir, folder)
            
            # Intentar primero con padding
            fp = os.path.join(folder_path, f"{window_padded}.png")
            
            if not os.path.exists(fp):
                # Buscar versiones aumentadas: 00902_Aug1.png, 00902_Aug2.png, etc.
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

                if pres == 1: 
                    pos_count += 1
                else: 
                    neg_count += 1
                    
                found = True
            except Exception as e:
                print(f"❌ Error cargando {fp}: {e}")
                continue

            break

        # if max_samples_per_class:
        #     if pos_count >= max_samples_per_class and neg_count >= max_samples_per_class:
        #         break

    print(f"\n📈 RESUMEN FINAL:")
    print(f"   ✅ Patches positivos: {pos_count}")
    print(f"   ✅ Patches negativos: {neg_count}")
    print(f"   📊 Ratio pos/neg: {pos_count/(neg_count+1e-8):.2f}")

    if pos_count == 0:
        print("⚠️ WARNING: No se cargaron patches positivos!")
        
    images = np.array(images)
    labels = np.array(labels)

    idxs = np.arange(len(labels))
    np.random.shuffle(idxs)

    return images[idxs], labels[idxs]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    X, y = build_annotated_arrays(
        ANNOTATED_ROOT, PATCH_EXCEL,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        image_size=IMAGE_SIZE
    )
    print("Cargados:", X.shape, np.unique(y, return_counts=True))
    print("DEBUG — Num samples:", len(y))
    print("DEBUG — Unique labels:", np.unique(y, return_counts=True))

    dataset = TripletDataset(X, y, transform=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = OneShotCNNNet(
        {'num_input_channels': 3},
        {'dim': 2, 'drop_rate': 0.3, 'block_configs': [[32, 32], [64, 64], [128, 128]]},
        {'prct': [1], 'hidden': [128], 'dropout': 0.5}
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = TripletLoss(margin=1.0)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        batches = 0

        for anchor, positive, negative, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            emb_a = model.feature_extraction(anchor)
            emb_p = model.feature_extraction(positive)
            emb_n = model.feature_extraction(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        print(f"Epoch {epoch+1} — Loss: {total_loss / batches:.6f}")

    os.makedirs(os.path.dirname(OUTPUT_BACKBONE), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_BACKBONE)
    print("Backbone guardado en:", OUTPUT_BACKBONE)


if __name__ == "__main__":
    main()
