import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch import optim

from Sistema2.Attention.patient_dataset import PatientLatentDataset
from Sistema2.Attention.patient_model import PatientAttentionModel

def collate_patient(batch):
    """
    batch: lista de (feats, y, codi), donde feats tiene tamaño (Ni, D) variable.
    Devuelve listas, no tensor apilado, porque Ni varía.
    """
    feats_list = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    codis = [b[2] for b in batch]
    return feats_list, labels, codis

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Sistema2/Triplets"
    feats_path  = os.path.join(base_dir, "latent_features.npy")
    labels_path = os.path.join(base_dir, "latent_labels.npy")
    annotated_csv_path = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/DatasetsTrain/AnnotatedTrain.csv"
    patient_csv_path   = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Datasets/PatientDiagnosis.csv"

    # Dataset de pacientes
    full_ds = PatientLatentDataset(feats_path, labels_path, annotated_csv_path, patient_csv_path, task="binary")

    # Split pacientes train/val (p.ej. 80/20)
    val_frac = 0.2
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_frac))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=4, collate_fn=collate_patient)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=4, collate_fn=collate_patient)

    model = PatientAttentionModel().to(device)
    # cargar pesos del embed preentrenado
    model.embed.load_state_dict(torch.load("/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Sistema2/Triplets/triplet_fc_512.pth", map_location="cpu"))
    # opcional: congelar embed al principio
    for p in model.embed.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3)

    max_epochs = 50
    best_val_loss = float("inf")
    best_state = None
    patience = 10
    no_impr = 0

    for epoch in range(max_epochs):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for feats_list, labels, _ in train_loader:
            # batch_size=1 → una lista de longitud 1
            feats = feats_list[0].to(device)  # (N_patches_i, D)
            labels = labels.to(device)        # (1,)

            logits, _ = model(feats)          # (1, 2)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ---- VAL ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats_list, labels, _ in val_loader:
                feats = feats_list[0].to(device)
                labels = labels.to(device)

                logits, _ = model(feats)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

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
                print("Early stopping in patient model.", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Sistema2/Attention/patient_attention_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Saved patient model to:", save_path)

if __name__ == "__main__":
    main()