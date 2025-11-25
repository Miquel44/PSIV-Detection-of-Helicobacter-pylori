#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema 2: Patch Classification + Attention-based Patient Diagnosis
Arquitectura: AutoEncoder -> Patch Classifier -> Attention Aggregation -> Patient Diagnosis
Usa la clase Attention de AttentionUnits.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import os
import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# ==================== GESTIÓN DE PATHS (SIGUIENDO AEExample_Script.py) ====================
current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent  # Sistema2
project_root = current_dir.parent  # PSIV-Detection-of-Helicobacter-pylori

# Rutas a módulos
autoencoder_path = project_root / 'AutoEncoder'
attention_path = current_dir / 'Attention'

sys.path.insert(0, str(autoencoder_path))
sys.path.insert(0, str(attention_path))

# Importar módulos
try:
    from Models.AEmodels import AutoEncoderCNN

    print("✓ AutoEncoderCNN importado correctamente")
except ImportError as e:
    print(f"✗ Error importando AutoEncoderCNN: {e}")
    sys.exit(1)

try:
    from AttentionUnits import Attention

    print("✓ Attention importado correctamente")
except ImportError as e:
    print(f"✗ Error importando Attention: {e}")
    sys.exit(1)

# ==================== PATHS DE DATOS (SIGUIENDO AEExample_Script.py) ====================
# Carpeta para resultados
RESULTS_DIR = 'Sistema2_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Rutas de datos (siguiendo la estructura de AEExample_Script.py)
DATA_ROOT = project_root / "Data"
DATASETS_ROOT = project_root / "Datasets"

ANNOTATED_CSV = "Data/annotated.csv"
PATIENT_DIAGNOSIS_CSV = "Data/PatientDiagnosis_AE.csv"

print(f"\n=== PATHS DE DATOS ===")
print(f"Data root: {DATA_ROOT}")
print(f"Annotated CSV: {ANNOTATED_CSV}")
print(f"Patient Diagnosis CSV: {PATIENT_DIAGNOSIS_CSV}")


# ==================== CONFIGURACIONES ====================

def AEConfigs(Config):
    """Configuraciones del AutoEncoder (idéntico a AEExample_Script.py)"""
    if Config == '1':
        net_paramsEnc = {
            'block_configs': [[32, 32], [64, 64]],
            'stride': [[1, 2], [1, 2]],
        }
        net_paramsDec = {
            'block_configs': [[64, 32], [32, 3]],  # 3 = num_input_channels
            'stride': [[1, 2], [1, 2]]
        }
        inputmodule_paramsDec = {
            'num_input_channels': net_paramsEnc['block_configs'][-1][-1]
        }
    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


inputmodule_paramsEnc = {
    'num_input_channels': 3
}


# ==================== MODELOS ====================

class PatchClassifier(nn.Module):
    """Clasificador binario de patches usando embeddings del AutoEncoder"""

    def __init__(self, embedding_dim, hidden_dim=256, num_classes=2):
        super().__init__()
        self.fc_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, embeddings):
        return self.fc_network(embeddings)


class AttentionPatientDiagnosis(nn.Module):
    """Modelo de diagnóstico por paciente usando Attention de AttentionUnits.py"""

    def __init__(self, embedding_dim, attention_dim=128, num_attention_branches=1):
        super().__init__()

        attention_params = {
            'in_features': embedding_dim,
            'decom_space': attention_dim,
            'ATTENTION_BRANCHES': num_attention_branches
        }

        self.attention = Attention(attention_params)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * num_attention_branches, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, patch_embeddings):
        H = patch_embeddings.unsqueeze(0)
        Z, A = self.attention(H)
        Z = Z.view(-1)
        diagnosis = self.classifier(Z)
        return diagnosis, A.squeeze(0)


# ==================== DATASETS ====================

class PatchDataset(Dataset):
    """Dataset para patches individuales con sus labels"""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['PATH']
        label = row['PRESENCE']

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), label, img_path


class PatientDataset(Dataset):
    """Dataset agrupado por paciente para diagnosis"""

    def __init__(self, patient_data, embeddings_dict, labels_dict):
        self.patient_ids = list(patient_data.keys())
        self.patient_data = patient_data
        self.embeddings_dict = embeddings_dict
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patch_paths = self.patient_data[patient_id]

        embeddings = []
        for path in patch_paths:
            if path in self.embeddings_dict:
                embeddings.append(self.embeddings_dict[path])

        if len(embeddings) == 0:
            embeddings.append(torch.zeros(512))

        embeddings = torch.stack(embeddings)
        label = self.labels_dict[patient_id]

        return embeddings, label, patient_id


# ==================== FUNCIONES AUXILIARES ====================

def load_data():
    """Carga datos desde los CSV pre-generados por 2_crear_csv.py"""

    # 1. Raíz del proyecto (donde está el código)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    # 2. Rutas a los CSV pre-procesados (en el proyecto)
    annotated_csv = project_root / "Data" / "annotated.csv"
    diagnosis_csv = project_root / "Datasets" / "PatientDiagnosis.csv"

    # 3. Ruta base del DATASET (donde están las imágenes reales)
    DATASET_ROOT = Path("/export/fhome/maed/HelicoDataSet")
    base_img_path = DATASET_ROOT / "CrossValidation" / "Annotated"

    print(f"Buscando archivos en:")
    print(f"  - Anotaciones: {annotated_csv}")
    print(f"  - Diagnósticos: {diagnosis_csv}")
    print(f"  - Imágenes base: {base_img_path}")

    # 4. Verificar si existen los CSVs
    if not annotated_csv.exists():
        print(f"\n⚠ ERROR: No existe {annotated_csv}")
        print("Cargando desde el Excel y generando rutas manualmente...")

        # Fallback: cargar desde Excel
        excel_path = project_root / "Datasets" / "HP_WSI-CoordAllAnnotatedPatches.xlsx"

        df_annotated = pd.read_excel(excel_path)
        print(f"✓ Excel cargado: {len(df_annotated)} filas")
        print(f"  Columnas disponibles: {df_annotated.columns.tolist()}")

        # Filtrar inciertos (Presence = 0)
        df_annotated = df_annotated[df_annotated['Presence'] != 0].copy()
        print(f"✓ Filtrados patches inciertos: {len(df_annotated)} de {len(df_annotated)} restantes")

        # Crear CODI
        df_annotated['CODI'] = df_annotated['Pat_ID']

        # Construir PATH con formato correcto (5 dígitos + ruta CORRECTA)
        def build_image_path(row):
            patient_folder = f"{row['Pat_ID']}_{row['Section_ID']}"
            window_id_padded = str(row['Window_ID']).zfill(5)
            return str(base_img_path / patient_folder / f"{window_id_padded}.png")

        df_annotated['PATH'] = df_annotated.apply(build_image_path, axis=1)

        # Mapear Presence (1 = H.pylori, -1 = sano)
        df_annotated['PRESENCE'] = df_annotated['Presence'].map({1: 1, -1: 0})
        df_annotated = df_annotated[df_annotated['PRESENCE'].notna()].copy()

    else:
        # Cargar desde CSV pero CORREGIR las rutas
        df_annotated = pd.read_csv(annotated_csv)
        print(f"✓ CSV cargado: {len(df_annotated)} filas")

        # IMPORTANTE: Reemplazar la ruta base incorrecta
        df_annotated['PATH'] = df_annotated['PATH'].str.replace(
            '/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Data',
            str(DATASET_ROOT),
            regex=False
        )

    # 5. Verificar que los archivos existen
    initial_len = len(df_annotated)
    df_annotated['EXISTS'] = df_annotated['PATH'].apply(os.path.exists)

    print(f"\nEjemplos de rutas (primeras 3):")
    for i, (path, exists) in enumerate(zip(df_annotated['PATH'].head(3),
                                           df_annotated['EXISTS'].head(3))):
        status = '✓ Existe' if exists else '✗ NO EXISTE'
        print(f"  {i + 1}. {path}")
        print(f"      {status}")

    df_annotated = df_annotated[df_annotated['EXISTS']].drop(columns=['EXISTS'])
    print(f"✓ Imágenes encontradas: {len(df_annotated)} de {initial_len}")

    if len(df_annotated) == 0:
        print("ERROR: No se encontraron imágenes en disco.")
        print(f"Verifica que las imágenes existen en {base_img_path}")
        sys.exit(1)

    # 6. Cargar diagnósticos
    try:
        df_diagnosis = pd.read_csv(diagnosis_csv)
        print(f"✓ Diagnósticos: {len(df_diagnosis)} pacientes")
        print(f"  Columnas: {df_diagnosis.columns.tolist()}")
    except Exception as e:
        print(f"ERROR leyendo {diagnosis_csv}: {e}")
        sys.exit(1)

    if 'CODI' not in df_diagnosis.columns or 'DIAGNOSIS' not in df_diagnosis.columns:
        print("ERROR: PatientDiagnosis_AE.csv debe tener columnas CODI y DIAGNOSIS")
        sys.exit(1)

    # 7. Estadísticas
    print(f"\nDistribución de patches:")
    print(f"  Sanos (PRESENCE=0): {sum(df_annotated['PRESENCE'] == 0)}")
    print(f"  H. pylori (PRESENCE=1): {sum(df_annotated['PRESENCE'] == 1)}")

    print(f"\nDistribución de pacientes:")
    print(f"  Negativos (DIAGNOSIS=0): {sum(df_diagnosis['DIAGNOSIS'] == 0)}")
    print(f"  Positivos (DIAGNOSIS=1): {sum(df_diagnosis['DIAGNOSIS'] == 1)}")

    return df_annotated.reset_index(drop=True), df_diagnosis.reset_index(drop=True)


def extract_embeddings_batch(autoencoder, dataloader, device):
    """Extrae embeddings del espacio latente del AutoEncoder"""
    autoencoder.eval()
    embeddings_dict = {}

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            _, embeddings = autoencoder(images)
            embeddings = embeddings.view(embeddings.size(0), -1)

            for i, path in enumerate(paths):
                embeddings_dict[path] = embeddings[i].cpu()

    return embeddings_dict


def group_patches_by_patient(df_annotated, df_diagnosis):
    """Agrupa patches por paciente y obtiene labels"""
    patient_patches = defaultdict(list)
    patient_labels = {}

    for idx, row in df_annotated.iterrows():
        patient_id = row['CODI']
        patient_patches[patient_id].append(row['PATH'])

    for _, row in df_diagnosis.iterrows():
        patient_id = row['CODI']
        patient_labels[patient_id] = row['DIAGNOSIS']

    common_patients = set(patient_patches.keys()) & set(patient_labels.keys())
    patient_patches = {k: v for k, v in patient_patches.items() if k in common_patients}
    patient_labels = {k: v for k, v in patient_labels.items() if k in common_patients}

    return patient_patches, patient_labels


def split_patients(patient_patches, patient_labels, test_size=0.2, val_size=0.1):
    """Split pacientes en train/val/test"""
    patient_ids = list(patient_patches.keys())
    labels = [patient_labels[pid] for pid in patient_ids]

    train_val_ids, test_ids = train_test_split(
        patient_ids, test_size=test_size, stratify=labels, random_state=42
    )

    train_val_labels = [patient_labels[pid] for pid in train_val_ids]
    val_size_adjusted = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size_adjusted,
        stratify=train_val_labels, random_state=42
    )

    return train_ids, val_ids, test_ids


# ==================== FASE 1: PATCH CLASSIFICATION ====================

def train_patch_classifier(autoencoder, df_annotated, device, epochs=20):
    """Entrena el clasificador de patches"""
    print("\n" + "=" * 50)
    print("FASE 1: PATCH CLASSIFICATION")
    print("=" * 50)

    dataset = PatchDataset(df_annotated)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    print("\nExtrayendo embeddings del AutoEncoder...")
    embeddings_dict = extract_embeddings_batch(autoencoder, dataloader, device)

    embeddings_list = []
    labels_list = []
    for idx, row in df_annotated.iterrows():
        path = row['PATH']
        if path in embeddings_dict:
            embeddings_list.append(embeddings_dict[path])
            labels_list.append(row['PRESENCE'])

    embeddings = torch.stack(embeddings_list)
    labels = torch.tensor(labels_list)

    embedding_dim = embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")

    indices = torch.randperm(len(embeddings))
    split_idx = int(0.8 * len(embeddings))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    train_emb, train_labels = embeddings[train_idx], labels[train_idx]
    test_emb, test_labels = embeddings[test_idx], labels[test_idx]

    classifier = PatchClassifier(embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0

    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0
        batch_size = 64
        indices = torch.randperm(len(train_emb))

        for i in range(0, len(train_emb), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_emb = train_emb[batch_idx].to(device)
            batch_labels = train_labels[batch_idx].to(device)

            optimizer.zero_grad()
            outputs = classifier(batch_emb)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        classifier.eval()
        with torch.no_grad():
            test_outputs = classifier(test_emb.to(device))
            preds = torch.argmax(test_outputs, dim=1).cpu()
            acc = accuracy_score(test_labels, preds)

        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), f"{RESULTS_DIR}/patch_classifier.pth")

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f}")

    print(f"\nBest Patch Classification Accuracy: {best_acc:.4f}")

    return classifier, embeddings_dict


# ==================== FASE 2: PATIENT DIAGNOSIS ====================

def train_patient_diagnosis(classifier, embeddings_dict, patient_patches,
                            patient_labels, device, epochs=50):
    """Entrena el modelo de diagnóstico por paciente"""
    print("\n" + "=" * 50)
    print("FASE 2: PATIENT DIAGNOSIS CON ATTENTION")
    print("=" * 50)

    train_ids, val_ids, test_ids = split_patients(patient_patches, patient_labels)

    print(f"Train patients: {len(train_ids)}")
    print(f"Val patients: {len(val_ids)}")
    print(f"Test patients: {len(test_ids)}")

    train_patient_data = {pid: patient_patches[pid] for pid in train_ids}
    val_patient_data = {pid: patient_patches[pid] for pid in val_ids}
    test_patient_data = {pid: patient_patches[pid] for pid in test_ids}

    train_dataset = PatientDataset(train_patient_data, embeddings_dict, patient_labels)
    val_dataset = PatientDataset(val_patient_data, embeddings_dict, patient_labels)
    test_dataset = PatientDataset(test_patient_data, embeddings_dict, patient_labels)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sample_emb = next(iter(train_loader))[0]
    embedding_dim = sample_emb.shape[-1]

    patient_model = AttentionPatientDiagnosis(
        embedding_dim, attention_dim=128, num_attention_branches=1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(patient_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=5)

    best_val_acc = 0

    for epoch in range(epochs):
        patient_model.train()
        epoch_loss = 0

        for embeddings, label, patient_id in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            embeddings = embeddings.squeeze(0).to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs, attention = patient_model(embeddings)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        patient_model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for embeddings, label, patient_id in val_loader:
                embeddings = embeddings.squeeze(0).to(device)
                label = label.to(device)

                outputs, _ = patient_model(embeddings)
                pred = torch.argmax(outputs, dim=0)
                val_correct += (pred == label).sum().item()
                val_total += 1

        val_acc = val_correct / val_total if val_total > 0 else 0
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(patient_model.state_dict(), f"{RESULTS_DIR}/patient_model.pth")

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    return patient_model


# ==================== MAIN ====================

def main():
    """Pipeline completo del Sistema 2"""
    print("=" * 60)
    print("SISTEMA 2: ATTENTION-BASED PATIENT DIAGNOSIS")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsando dispositivo: {device}")

    # Cargar datos
    df_annotated, df_diagnosis = load_data()

    # Cargar AutoEncoder
    print("\nCargando AutoEncoder...")
    Config = '1'
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)

    autoencoder = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                                 inputmodule_paramsDec, net_paramsDec)

    ae_path = "AutoEncoder/Models/Trained/AE_Config1_fold1_best.pth"
    if os.path.exists(ae_path):
        autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
        print(f"✓ AutoEncoder cargado desde {ae_path}")
    else:
        print(f"⚠ No se encontró {ae_path}")

    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # Fase 1
    patch_classifier, embeddings_dict = train_patch_classifier(
        autoencoder, df_annotated, device, epochs=20
    )

    # Agrupar por paciente
    patient_patches, patient_labels = group_patches_by_patient(df_annotated, df_diagnosis)

    # Fase 2
    patient_model = train_patient_diagnosis(
        patch_classifier, embeddings_dict, patient_patches,
        patient_labels, device, epochs=50
    )

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()