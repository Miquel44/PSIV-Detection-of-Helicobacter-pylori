import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd

from Sistema2.Triplets.datasets import TripletDataset
from Sistema2.Triplets.triplet_loss import TripletLoss

LATENT_DIM = 262144   # de check_latent_dim
EMBED_DIM  = 512

class PatchEmbeddingFC(nn.Module):
    def __init__(self, in_dim=LATENT_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, embed_dim)   # una sola FC sencilla

    def forward(self, x):
        e = self.fc(x)
        e = nn.functional.normalize(e, dim=1)
        return e

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Sistema2/Triplets"
    feats_path  = os.path.join(base_dir, "latent_features.npy")
    labels_path = os.path.join(base_dir, "latent_labels.npy")

    # features y labels
    X = np.load(feats_path).astype(np.float32)   # (N, LATENT_DIM)
    y = np.load(labels_path).astype(np.int64)    # (N,)

    # ----------------------------
    # 1) Cargar CSV para tener CODI por patch
    # ----------------------------
    csv_path = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/DatasetsTrain/AnnotatedTrain.csv"
    df = pd.read_csv(csv_path)

    assert len(df) == X.shape[0], "N filas de CSV y N de latent_features no coinciden"

    codis = df["CODI"].values
    unique_codis = np.unique(codis)

    # split por paciente (p.ej. 90% train, 10% val)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_codis)

    val_frac = 0.2
    n_val_pac = max(1, int(len(unique_codis) * val_frac))
    val_codis = set(unique_codis[:n_val_pac])
    train_codis = set(unique_codis[n_val_pac:])

    train_mask = np.array([c in train_codis for c in codis])
    val_mask   = np.array([c in val_codis   for c in codis])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f"Train patches: {X_train.shape[0]}, Val patches: {X_val.shape[0]}")

    full_train_ds = TripletDataset(X_train, y_train)
    full_val_ds   = TripletDataset(X_val,   y_val)

    print("Train labels:", np.bincount(y_train))
    print("Val labels:",   np.bincount(y_val))

    train_loader = DataLoader(full_train_ds, batch_size=64, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(full_val_ds,   batch_size=64, shuffle=False, num_workers=4)

    model = PatchEmbeddingFC(in_dim=LATENT_DIM, embed_dim=EMBED_DIM).to(device)
    criterion = TripletLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    max_epochs     = 100
    patience       = 10
    best_val_loss  = float("inf")
    epochs_no_impr = 0
    best_state     = None

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}", flush=True)

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        n = 0
        for anchor, positive, negative, _ in train_loader:
            anchor   = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            ea = model(anchor)
            ep = model(positive)
            en = model(negative)

            loss = criterion(ea, ep, en)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * anchor.size(0)
            n += anchor.size(0)

        train_loss = running_loss / n

        # ---- VALIDATION ----
        model.eval()
        val_running = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for anchor, positive, negative, _ in val_loader:
                anchor   = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                ea = model(anchor)
                ep = model(positive)
                en = model(negative)

                vloss = criterion(ea, ep, en)
                val_running += vloss.item() * anchor.size(0)
                n_val_samples += anchor.size(0)

        val_loss = val_running / n_val_samples

        print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}", flush=True)

        # ---- EARLY STOPPING ----
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_impr = 0
            best_state = model.state_dict()
        else:
            epochs_no_impr += 1
            print(f"  No improvement for {epochs_no_impr} epoch(s)", flush=True)
            if epochs_no_impr >= patience:
                print("Early stopping triggered.", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = os.path.join(base_dir, "triplet_fc_512.pth")
    torch.save(model.state_dict(), save_path)
    print("Saved triplet head to:", save_path)

if __name__ == "__main__":
    main()