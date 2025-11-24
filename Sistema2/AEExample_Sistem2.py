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

# Añadir paths necesarios
sys.path.append('../AutoEncoder')
sys.path.append('Attention')

from Models.AEmodels import AutoEncoderCNN
from AttentionUnits import Attention  # USAR LA CLASE EXISTENTE

# Carpeta para resultados
RESULTS_DIR = 'Sistema2_Results'

# ==================== CONFIGURACIONES ====================

def AEConfigs(Config):
    """Configuraciones del AutoEncoder (sin modificar)"""
    if Config == '1':
        net_paramsEnc = {
            'block_configs': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
            'stride': [[1, 0], [1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
            'num_blocks_list': [2, 2, 3, 3, 3],
            'expansion': 1,
            'block_type': 'Basic'
        }
        net_paramsDec = {
            'in_channels_list': [512, 512, 256, 128, 64],
            'out_channels_list': [512, 256, 128, 64, 64],
            'stride_list': [1, 1, 1, 1, 1],
            'num_blocks_list': [3, 3, 3, 2, 2],
            'expansion': 1,
            'block_type': 'Basic'
        }
        inputmodule_paramsDec = {
            'num_filters': 3,
            'filter_size': [5, 5],
            'stride': [2, 2],
            'padding': [2, 2]
        }
    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


inputmodule_paramsEnc = {
    'num_filters': 64,
    'filter_size': [5, 5],
    'stride': [2, 2],
    'padding': [2, 2],
    'dilation': [1, 1]
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
    """
    Modelo de diagnóstico por paciente USANDO la clase Attention de AttentionUnits.py
    """

    def __init__(self, embedding_dim, attention_dim=128, num_attention_branches=1):
        super().__init__()

        # Parámetros para la clase Attention de AttentionUnits.py
        attention_params = {
            'in_features': embedding_dim,
            'decom_space': attention_dim,
            'ATTENTION_BRANCHES': num_attention_branches
        }

        # USAR LA CLASE ATTENTION DE AttentionUnits.py
        self.attention = Attention(attention_params)

        # MLP para diagnóstico final
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * num_attention_branches, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, patch_embeddings):
        """
        Args:
            patch_embeddings: [num_patches, embedding_dim]
        Returns:
            diagnosis: [2] (logits para clasificación binaria)
            attention: [num_patches] (pesos de attention)
        """
        # Añadir dimensión batch para la clase Attention
        H = patch_embeddings.unsqueeze(0)  # [1, num_patches, embedding_dim]

        # Context Vector + Attention weights (USAR AttentionUnits.py)
        Z, A = self.attention(H)  # Z: [ATTENTION_BRANCHES, M], A: [ATTENTION_BRANCHES, NV]

        # Flatten context vector
        Z = Z.view(-1)  # [ATTENTION_BRANCHES * M]

        # Diagnóstico
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
    """Carga los datos anotados y diagnósticos"""
    annotated_path = "../Datasets/HP_WSI-CoordAllAnnotatedPatches.xlsx"
    df_annotated = pd.read_excel(annotated_path)

    base_path = "../Data/Cropped"
    df_annotated['PATH'] = df_annotated.apply(
        lambda row: os.path.join(base_path, row['CODI'],
                                 f"{row['CODI']}_{row['NUMERO']}.png"),
        axis=1
    )

    df_annotated = df_annotated[df_annotated['PATH'].apply(os.path.exists)]

    diagnosis_path = "../Datasets/PatientDiagnosis.csv"
    df_diagnosis = pd.read_csv(diagnosis_path)

    return df_annotated, df_diagnosis


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
        diagnosis = row.iloc[1]
        patient_labels[patient_id] = 0 if diagnosis == 'NEGATIVA' else 1

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
    """Entrena el clasificador de patches usando embeddings del AutoEncoder"""
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
    train_losses = []
    val_accs = []

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

        train_losses.append(epoch_loss / (len(train_emb) // batch_size))
        val_accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), "patch_classifier_phase2.pth")

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f}")

    plot_patch_training(train_losses, val_accs)

    classifier.eval()
    with torch.no_grad():
        test_outputs = classifier(test_emb.to(device))
        preds = torch.argmax(test_outputs, dim=1).cpu()

    print("\n=== RESULTADOS PATCH CLASSIFICATION ===")
    print(classification_report(test_labels, preds, target_names=['Negativo', 'Positivo']))

    cm = confusion_matrix(test_labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Patch Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{RESULTS_DIR}/patch_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nBest Patch Classification Accuracy: {best_acc:.4f}")

    return classifier, embeddings_dict


# ==================== FASE 2: PATIENT DIAGNOSIS ====================

def train_patient_diagnosis(classifier, embeddings_dict, patient_patches,
                            patient_labels, device, epochs=50):
    """Entrena el modelo de diagnóstico por paciente usando Attention de AttentionUnits.py"""
    print("\n" + "=" * 50)
    print("FASE 2: PATIENT DIAGNOSIS CON ATTENTION (AttentionUnits.py)")
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

    # Obtener dimensión de embeddings
    sample_emb = next(iter(train_loader))[0]
    embedding_dim = sample_emb.shape[-1]
    print(f"Embedding dimension: {embedding_dim}")

    patient_model = AttentionPatientDiagnosis(
        embedding_dim, attention_dim=128, num_attention_branches=1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(patient_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=5)

    best_val_acc = 0
    train_losses = []
    val_accs = []
    attention_history = []

    for epoch in range(epochs):
        patient_model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for embeddings, label, patient_id in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            embeddings = embeddings.squeeze(0).to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs, attention = patient_model(embeddings)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = torch.argmax(outputs, dim=0)
            train_correct += (pred == label).sum().item()
            train_total += 1

        train_acc = train_correct / train_total if train_total > 0 else 0
        avg_loss = epoch_loss / train_total if train_total > 0 else 0

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

        train_losses.append(avg_loss)
        val_accs.append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(patient_model.state_dict(), "patient_model_phase3.pth")

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    plot_patient_training(train_losses, val_accs)

    print("\n=== EVALUACIÓN EN TEST SET ===")
    patient_model.load_state_dict(torch.load("patient_model_phase3.pth"))
    patient_model.eval()

    test_preds = []
    test_labels_list = []
    test_probs = []
    test_attention = []

    with torch.no_grad():
        for embeddings, label, patient_id in test_loader:
            embeddings = embeddings.squeeze(0).to(device)
            outputs, attention = patient_model(embeddings)

            prob = F.softmax(outputs, dim=0)[1].item()
            pred = torch.argmax(outputs, dim=0).item()

            test_preds.append(pred)
            test_labels_list.append(label.item())
            test_probs.append(prob)

            attention_weights = attention.cpu().numpy()
            if attention_weights.ndim > 1:
                attention_weights = attention_weights[0]

            test_attention.append({
                'patient': patient_id[0],
                'attention': attention_weights,
                'label': label.item(),
                'pred': pred
            })

    print(classification_report(test_labels_list, test_preds,
                                target_names=['Negativo', 'Positivo']))

    cm = confusion_matrix(test_labels_list, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Patient Diagnosis')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{RESULTS_DIR}/patient_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    fpr, tpr, _ = roc_curve(test_labels_list, test_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Patient Diagnosis')
    plt.legend(loc="lower right")
    plt.savefig(f'{RESULTS_DIR}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_attention_analysis(test_attention)

    print(f"\nTest Accuracy: {accuracy_score(test_labels_list, test_preds):.4f}")
    print(f"Test AUC: {roc_auc:.4f}")

    return patient_model, test_attention


# ==================== VISUALIZACIONES ====================

def plot_patch_training(losses, accs):
    """Gráficas de entrenamiento de patch classifier"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Patch Classifier - Training Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(accs, label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Patch Classifier - Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/patch_training.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_patient_training(losses, accs):
    """Gráficas de entrenamiento de patient diagnosis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Patient Diagnosis - Training Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(accs, label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Patient Diagnosis - Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/patient_training.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_analysis(attention_history):
    """Análisis visual de los attention weights"""
    if len(attention_history) == 0:
        return

    fig = plt.figure(figsize=(20, 12))

    # 1. Distribución de attention por paciente
    ax1 = plt.subplot(3, 3, 1)
    for i, item in enumerate(attention_history[:5]):
        ax1.plot(item['attention'], label=f"Patient {item['patient']}", alpha=0.7)
    ax1.set_xlabel('Patch Index')
    ax1.set_ylabel('Attention Weight')
    ax1.set_title('Attention Weights - Sample Patients')
    ax1.legend()
    ax1.grid(True)

    # 2. Heatmap de attention
    ax2 = plt.subplot(3, 3, 2)
    sample_size = min(20, len(attention_history))
    attention_matrix = [item['attention'] for item in attention_history[:sample_size]]
    max_len = max([len(a) for a in attention_matrix])
    attention_padded = np.array([np.pad(a, (0, max_len - len(a))) for a in attention_matrix])
    sns.heatmap(attention_padded, cmap='viridis', ax=ax2, cbar_kws={'label': 'Attention'})
    ax2.set_title('Attention Heatmap (Top 20 Patients)')
    ax2.set_xlabel('Patch Index')
    ax2.set_ylabel('Patient Index')

    # 3. Attention vs Label
    ax3 = plt.subplot(3, 3, 3)
    neg_att = [np.mean(item['attention']) for item in attention_history if item['label'] == 0]
    pos_att = [np.mean(item['attention']) for item in attention_history if item['label'] == 1]
    ax3.boxplot([neg_att, pos_att], labels=['Negativo', 'Positivo'])
    ax3.set_ylabel('Mean Attention Weight')
    ax3.set_title('Mean Attention por Diagnóstico')
    ax3.grid(True)

    # 4. Distribución de patches
    ax4 = plt.subplot(3, 3, 4)
    num_patches = [len(item['attention']) for item in attention_history]
    ax4.hist(num_patches, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Number of Patches')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Patches per Patient')
    ax4.grid(True)

    # 5. Max attention vs Label
    ax5 = plt.subplot(3, 3, 5)
    neg_max = [np.max(item['attention']) for item in attention_history if item['label'] == 0]
    pos_max = [np.max(item['attention']) for item in attention_history if item['label'] == 1]
    ax5.boxplot([neg_max, pos_max], labels=['Negativo', 'Positivo'])
    ax5.set_ylabel('Max Attention Weight')
    ax5.set_title('Max Attention por Diagnóstico')
    ax5.grid(True)

    # 6. Attention variance
    ax6 = plt.subplot(3, 3, 6)
    neg_var = [np.std(item['attention']) for item in attention_history if item['label'] == 0]
    pos_var = [np.std(item['attention']) for item in attention_history if item['label'] == 1]
    ax6.boxplot([neg_var, pos_var], labels=['Negativo', 'Positivo'])
    ax6.set_ylabel('Attention Std Dev')
    ax6.set_title('Attention Variability por Diagnóstico')
    ax6.grid(True)

    # 7. Top-k attention concentration
    ax7 = plt.subplot(3, 3, 7)
    k = 5
    neg_topk = [np.sum(np.sort(item['attention'])[-k:]) for item in attention_history if item['label'] == 0]
    pos_topk = [np.sum(np.sort(item['attention'])[-k:]) for item in attention_history if item['label'] == 1]
    ax7.boxplot([neg_topk, pos_topk], labels=['Negativo', 'Positivo'])
    ax7.set_ylabel(f'Sum of Top-{k} Attention')
    ax7.set_title(f'Top-{k} Attention Concentration')
    ax7.grid(True)

    # 8. Attention distribution
    ax8 = plt.subplot(3, 3, 8)
    all_attention = np.concatenate([item['attention'] for item in attention_history])
    ax8.hist(all_attention, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax8.set_xlabel('Attention Weight')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Overall Attention Distribution')
    ax8.grid(True)

    # 9. Correctness vs attention
    ax9 = plt.subplot(3, 3, 9)
    correct_att = [np.mean(item['attention']) for item in attention_history
                   if item['label'] == item['pred']]
    wrong_att = [np.mean(item['attention']) for item in attention_history
                 if item['label'] != item['pred']]
    if len(wrong_att) > 0:
        ax9.boxplot([correct_att, wrong_att], labels=['Correct', 'Wrong'])
    else:
        ax9.boxplot([correct_att], labels=['Correct'])
    ax9.set_ylabel('Mean Attention')
    ax9.set_title('Attention vs Prediction Correctness')
    ax9.grid(True)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Análisis de Attention guardado en {RESULTS_DIR}/attention_analysis.png")


# ==================== MAIN ====================

def main():
    """Pipeline completo del Sistema 2"""
    print("=" * 60)
    print("SISTEMA 2: ATTENTION-BASED PATIENT DIAGNOSIS")
    print("Usando clase Attention de AttentionUnits.py")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsando dispositivo: {device}")

    # Crear carpeta específica para plots del Sistema 2
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"✓ Carpeta de resultados: {RESULTS_DIR}")

    print("\nCargando datos...")
    df_annotated, df_diagnosis = load_data()
    print(f"Patches anotados: {len(df_annotated)}")
    print(f"Pacientes con diagnóstico: {len(df_diagnosis)}")

    print("\nCargando AutoEncoder pre-entrenado...")
    Config = '1'
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)

    autoencoder = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                                 inputmodule_paramsDec, net_paramsDec)

    if os.path.exists("backbone_phase1.pth"):
        autoencoder.load_state_dict(torch.load("backbone_phase1.pth", map_location=device))
        print("✓ AutoEncoder cargado desde backbone_phase1.pth")
    else:
        print("⚠ No se encontró backbone_phase1.pth, usando modelo sin entrenar")

    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # Fase 1: Patch Classification
    patch_classifier, embeddings_dict = train_patch_classifier(
        autoencoder, df_annotated, device, epochs=20
    )

    print("\nAgrupando patches por paciente...")
    patient_patches, patient_labels = group_patches_by_patient(df_annotated, df_diagnosis)
    print(f"Total de pacientes: {len(patient_patches)}")

    # Fase 2: Patient Diagnosis con Attention
    patient_model, test_attention = train_patient_diagnosis(
        patch_classifier, embeddings_dict, patient_patches,
        patient_labels, device, epochs=50
    )

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print("\nArchivos generados:")
    print("  Modelos:")
    print("    - patch_classifier_phase2.pth")
    print("    - patient_model_phase3.pth")
    print(f"\n  Plots en {RESULTS_DIR}/:")
    print("    - patch_confusion_matrix.png")
    print("    - patient_confusion_matrix.png")
    print("    - roc_curve.png")
    print("    - attention_analysis.png")
    print("    - patch_training.png")
    print("    - patient_training.png")


if __name__ == "__main__":
    main()