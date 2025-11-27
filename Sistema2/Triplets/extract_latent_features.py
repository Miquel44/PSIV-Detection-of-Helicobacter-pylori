import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from AutoEncoderMetrics.Models.AEmodels import AutoEncoderCNN

class AnnotatedPatchDataset(Dataset):
    """
    Lee AnnotatedTrain.csv y devuelve imágenes 256x256 + label PRESENCE.
    """
    def __init__(self, csv_path, image_size=256):
        self.df = pd.read_csv(csv_path)
        self.paths = self.df["PATH"].values
        self.labels = self.df["PRESENCE"].values.astype(int)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return img, label

def build_encoder(ckpt_path):
    """
    Construye el AutoEncoderCNN con la misma configuración que usaste
    (Config '1', 256x256) y devuelve el encoder ya cargado.
    """
    # Esta configuración está fija según lo que has usado en el sistema 1
    inputmodule_paramsEnc = {
        'num_input_channels': 3,
        'n_feats_in': 256,
    }
    net_paramsEnc = {
        'block_configs': [[32, 32], [64, 64]],
        'stride': [[1, 2], [1, 2]],
    }

    inputmodule_paramsDec = {
        'num_input_channels': 64,   # típico: canales del último bloque encoder
        'n_feats_in': 64,           # tamaño espacial 64 (porque output es 64x64x64)
    }
    net_paramsDec = {
        'block_configs': [[64, 32], [32, inputmodule_paramsEnc['num_input_channels']]],
        'stride': [[1, 2], [1, 2]],
    }

    ae = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                        inputmodule_paramsDec, net_paramsDec)

    state = torch.load(ckpt_path, map_location="cpu")
    ae.load_state_dict(state)

    return ae.encoder

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_path = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/DatasetsTrain/AnnotatedTrain.csv"
    ckpt_path = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/Models/Trained/AE_Config1_best_MSELoss_256.pth"

    # 1) Dataset y loader
    dataset = AnnotatedPatchDataset(csv_path, image_size=256)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    # 2) Encoder del AE
    enc = build_encoder(ckpt_path)
    enc.to(device)
    enc.eval()

    # 3) Reservar arrays para features y labels
    n = len(dataset)
    latent_dim = 64 * 64 * 64  # lo que medimos: 262144
    feats = np.zeros((n, latent_dim), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)

    idx_base = 0
    with torch.no_grad():
        for imgs, batch_labels in tqdm(loader, desc="Extracting latent features"):
            bsz = imgs.size(0)
            imgs = imgs.to(device)

            z = enc(imgs)                   # (B, 64, 64, 64)
            z = z.view(bsz, -1).cpu().numpy()

            feats[idx_base:idx_base+bsz] = z
            labels[idx_base:idx_base+bsz] = batch_labels.numpy()
            idx_base += bsz

    out_dir = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Sistema2/Triplets"
    os.makedirs(out_dir, exist_ok=True)

    feats_path = os.path.join(out_dir, "latent_features.npy")
    labels_path = os.path.join(out_dir, "latent_labels.npy")

    np.save(feats_path, feats)
    np.save(labels_path, labels)

    print("Saved features to:", feats_path)
    print("Saved labels   to:", labels_path)

if __name__ == "__main__":
    main()