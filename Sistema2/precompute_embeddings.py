#!/usr/bin/env python3
# precompute_embeddings.py

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm
from AutoEncoder.Models.models import OneShotCNNNet


# ------------------------------------------------------------
#  CONFIGURACIÓN
# ------------------------------------------------------------
CROPPED_ROOT = "/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped/"
BACKBONE_CKPT = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/backbone_phase1.pth"
OUTPUT_NPZ = "./precomputed_patient_embeddings.npz"

IMAGE_SIZE = 256  # ← CAMBIADO: debe coincidir con Phase 1 y 2
MAX_PATCHES_PER_PATIENT = None  # ← CAMBIADO: cargar todos los patches
# ------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")

    # Cargar backbone
    print(f"\n🔧 Cargando backbone desde {BACKBONE_CKPT}")
    model = OneShotCNNNet(
        {'num_input_channels': 3},
        {'dim': 2, 'drop_rate': 0.3, 'block_configs': [[32, 32], [64, 64], [128, 128]]},
        {'prct': [1], 'hidden': [128], 'dropout': 0.5}
    )
    model.load_state_dict(torch.load(BACKBONE_CKPT, map_location=device))
    model.to(device)
    model.eval()

    # Verificar dimensión de embeddings
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        dummy_emb = model.feature_extraction(dummy_input)
        emb_dim = dummy_emb.shape[1]
        print(f"✅ Dimensión de embeddings: {emb_dim}")

    tf = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor()
    ])
    
    out_dict = {}
    total_patches = 0
    failed_patients = 0

    folders = sorted(os.listdir(CROPPED_ROOT))
    print(f"\n📂 Procesando {len(folders)} pacientes...")

    for folder in tqdm(folders):
        path = os.path.join(CROPPED_ROOT, folder)
        if not os.path.isdir(path):
            continue

        files = [f for f in os.listdir(path) if f.endswith(".png")]
        
        # Opcional: limitar patches por paciente si hay problemas de memoria
        if MAX_PATCHES_PER_PATIENT and len(files) > MAX_PATCHES_PER_PATIENT:
            files = list(np.random.choice(files, MAX_PATCHES_PER_PATIENT, replace=False))

        emb_list = []
        with torch.no_grad():
            for f in files:
                fp = os.path.join(path, f)
                try:
                    img = Image.open(fp).convert("RGB")
                    x = tf(img).unsqueeze(0).to(device)
                    e = model.feature_extraction(x).cpu().numpy()[0]
                    emb_list.append(e)
                except Exception as ex:
                    # Silenciosamente continuar si falla un patch
                    continue

        if len(emb_list) > 0:
            out_dict[folder] = np.array(emb_list)
            total_patches += len(emb_list)
        else:
            failed_patients += 1

    print(f"\n✅ Total pacientes procesados: {len(out_dict)}")
    print(f"✅ Total patches embeddings: {total_patches}")
    print(f"✅ Promedio patches/paciente: {total_patches/len(out_dict):.1f}")
    
    if failed_patients > 0:
        print(f"⚠️ Pacientes sin patches válidos: {failed_patients}")
    
    np.savez_compressed(OUTPUT_NPZ, **out_dict)
    print(f"\n💾 Embeddings guardados en: {OUTPUT_NPZ}")
    print(f"💾 Tamaño del archivo: {os.path.getsize(OUTPUT_NPZ) / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()