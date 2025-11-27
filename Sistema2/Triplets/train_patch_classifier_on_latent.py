import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
import pandas as pd

from Sistema2.Triplets.training_triplet import PatchEmbeddingFC, LATENT_DIM, EMBED_DIM

class LatentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

class PatchClassifier(nn.Module):
    def __init__(self, in_dim=LATENT_DIM, embed_dim=EMBED_DIM, n_classes=2):
        super().__init__()
        self.embed = PatchEmbeddingFC(in_dim=in_dim, embed_dim=embed_dim)
        self.cls = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            e = self.embed(x)
        logits = self.cls(e)
        return logits

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Sistema2/Triplets"
    feats_path  = os.path.join(base_dir, "latent_features.npy")
    labels_path = os.path.join(base_dir, "latent_labels.npy")
    triplet_fc_path = os.path.join(base_dir, "triplet_fc_512.pth")

    X = np.load(feats_path)
    y = np.load(labels_path)

    # ----------------------------
    # 1) Split por paciente usando AnnotatedTrain.csv
    # ----------------------------
    csv_path = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/DatasetsTrain/AnnotatedTrain.csv"
    df = pd.read_csv(csv_path)
    assert len(df) == X.shape[0], "N filas CSV y N latent_features no coinciden"

    codis = df["CODI"].values
    unique_codis = np.unique(codis)

    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_codis)

    val_frac = 0.1
    n_val_pac = max(1, int(len(unique_codis) * val_frac))
    val_codis = set(unique_codis[:n_val_pac])
    train_codis = set(unique_codis[n_val_pac:])

    train_mask = np.array([c in train_codis for c in codis])
    val_mask   = np.array([c in val_codis   for c in codis])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f"Train patches: {X_train.shape[0]}, Val patches: {X_val.shape[0]}")

    train_ds = LatentDataset(X_train, y_train)
    val_ds   = LatentDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    model = PatchClassifier(in_dim=LATENT_DIM, embed_dim=EMBED_DIM, n_classes=2).to(device)
    model.embed.load_state_dict(torch.load(triplet_fc_path, map_location="cpu"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.cls.parameters(), lr=1e-3)

    max_epochs = 50
    best_val_loss = float("inf")
    best_state = None
    patience = 10
    no_impr = 0

    for epoch in range(max_epochs):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, labels in train_loader:
            x = x.to(device)
            labels = labels.to(device)

            logits = model(x)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- VAL ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(device)
                labels = labels.to(device)

                logits = model(x)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * x.size(0)

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += x.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{max_epochs} "
              f"- Train loss: {train_loss:.4f}, acc: {train_acc:.3f} "
              f"| Val loss: {val_loss:.4f}, acc: {val_acc:.3f}",
              flush=True)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                print("Early stopping in classifier.", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = os.path.join(base_dir, "patch_classifier_from_triplet_512.pth")
    torch.save(model.state_dict(), save_path)
    print("Saved patch classifier to:", save_path)

if __name__ == "__main__":
    main()