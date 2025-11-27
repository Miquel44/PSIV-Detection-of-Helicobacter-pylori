import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import gc
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from torchvision import transforms

# Permitir cargar imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append('/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder')
sys.path.append('/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Sistema2/Attention')

from Models.AEmodels import AutoEncoderCNN
from AttentionUnits import GatedAttention, NeuralNetwork


def extract_features_from_dataframe(df, encoder, device, max_patches=None):
    """Extrae features para todos los pacientes del DataFrame"""

    print("\n[EXTRACT] Iniciando extracción de features...", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("[EXTRACT] Cargando diagnósticos de pacientes...", flush=True)
    patient_diagnosis_df = pd.read_csv(
        "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Datasets/PatientDiagnosis.csv"
    )
    patient_labels = dict(zip(
        patient_diagnosis_df['CODI'],
        (patient_diagnosis_df.iloc[:, 1] != 'NEGATIVA').astype(int)
    ))
    print(f"[EXTRACT] ✓ Cargadas etiquetas de {len(patient_labels)} pacientes", flush=True)

    patients_grouped = df.groupby('CODI')
    patient_features = []
    patient_labels_list = []

    encoder.eval()

    total_patients = len(patients_grouped)
    print(f"[EXTRACT] Procesando {total_patients} pacientes...", flush=True)

    skipped_patches = 0
    skipped_patients = 0

    for patient_idx, (patient_id, patient_patches) in enumerate(patients_grouped):

        print(f"[EXTRACT] Paciente {patient_idx + 1}/{total_patients} (ID: {patient_id}, "
              f"Patches: {len(patient_patches)})", flush=True)

        # ===== SALTAR PACIENTE B22-87 =====
        if patient_id == 'B22-87':
            print(f"  └─ ⚠ Paciente {patient_id} omitido manualmente (rutas inválidas)", flush=True)
            skipped_patients += 1
            continue
        # ==================================

        if patient_id not in patient_labels:
            print(f"  └─ ⚠ Paciente {patient_id} sin diagnóstico, saltado", flush=True)
            skipped_patients += 1
            continue

        if max_patches and len(patient_patches) > max_patches:
            patient_patches = patient_patches.sample(n=max_patches, random_state=42)

        features_list = []
        processed_in_patient = 0

        for patch_idx, (_, row) in enumerate(patient_patches.iterrows()):
            try:
                if not os.path.exists(row['PATH']):
                    skipped_patches += 1
                    continue

                img = Image.open(row['PATH']).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device, non_blocking=True)

                with torch.no_grad():
                    features = encoder(img_tensor)
                    features_pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                    features_flat = features_pooled.view(features_pooled.size(0), -1)
                    features_list.append(features_flat.squeeze(0).cpu())

                processed_in_patient += 1

                if (patch_idx + 1) % 50 == 0:
                    print(f"  └─ Procesados {patch_idx + 1}/{len(patient_patches)} patches", flush=True)

            except Exception as e:
                print(f"  └─ ERROR en patch {patch_idx}: {row['PATH']} - {str(e)[:100]}", flush=True)
                skipped_patches += 1
                continue

        print(f"  └─ ✓ Paciente {patient_id}: {processed_in_patient}/{len(patient_patches)} patches válidos", flush=True)

        if len(features_list) > 0:
            patient_feature_tensor = torch.stack(features_list)
            patient_features.append(patient_feature_tensor)
            patient_labels_list.append(patient_labels[patient_id])
        else:
            print(f"  └─ ⚠ Paciente {patient_id} sin patches válidos, saltado", flush=True)
            skipped_patients += 1

    print(f"[EXTRACT] ✓ Extracción completada: {len(patient_features)} pacientes procesados", flush=True)
    print(f"[EXTRACT] ⚠ Patches saltados: {skipped_patches}", flush=True)
    print(f"[EXTRACT] ⚠ Pacientes saltados: {skipped_patients}", flush=True)

    return patient_features, patient_labels_list


class PrecomputedPatientDataset(Dataset):
    def __init__(self, features_list, labels_list):
        self.features = features_list
        self.labels = labels_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


if __name__ == "__main__":
    print("="*80)
    print("SISTEMA 2: Attention-based Patient Diagnosis")
    print("="*80)

    print("\n[INIT] Configurando parámetros...", flush=True)
    num_folds = 3
    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    max_patches_per_patient = 100

    inputmodule_paramsEnc = {'num_input_channels': 3}
    net_paramsEnc = {
        'block_configs': [[32, 32], [64, 64]],
        'stride': [[1, 2], [1, 2]]
    }
    net_paramsDec = {
        'block_configs': [[64, 32], [32, 3]],
        'stride': [[1, 2], [1, 2]]
    }
    inputmodule_paramsDec = {'num_input_channels': 64}

    print("[INIT] ✓ Parámetros configurados", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] Usando: {device}", flush=True)

    print("\n[MODEL] Cargando AutoEncoder preentrenado...", flush=True)
    encoder_full = AutoEncoderCNN(
        inputmodule_paramsEnc, net_paramsEnc,
        inputmodule_paramsDec, net_paramsDec
    )

    encoder_path = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder/Models/Trained/AE_Config1_fold1_best_Linux.pth"
    print(f"[MODEL] Cargando pesos desde: {encoder_path}", flush=True)
    encoder_full.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder = encoder_full.encoder.to(device)
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    print("[MODEL] ✓ Encoder cargado y congelado correctamente", flush=True)

    print("\n[MODEL] Calculando dimensión de features...", flush=True)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = encoder(dummy_input)
        pooled_output = torch.nn.functional.adaptive_avg_pool2d(dummy_output, (1, 1))
        feature_dim = pooled_output.shape[1]

    print(f"[MODEL] ✓ Dimensión de features del encoder: {feature_dim}", flush=True)

    del encoder_full, dummy_input, dummy_output
    gc.collect()
    torch.cuda.empty_cache()
    print("[MODEL] ✓ Memoria del autoencoder liberada", flush=True)

    print("\n[DATA] Cargando dataset...", flush=True)
    df_cropped = pd.read_csv(
        "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Data/PatientDiagnosis_AE_Linux.csv"
    )
    patient_ids = df_cropped["CODI"].tolist()
    print(f"[DATA] ✓ Cargados {len(df_cropped)} patches de {len(set(patient_ids))} pacientes", flush=True)

    print(f"\n[FOLD] Iniciando {num_folds}-Fold Cross-Validation...", flush=True)
    gkf = GroupKFold(n_splits=num_folds)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df_cropped, groups=patient_ids)):
        print("\n" + "="*80)
        print(f"FOLD {fold_idx + 1}/{num_folds}")
        print("="*80)

        df_train = df_cropped.iloc[train_idx].reset_index(drop=True)
        df_val = df_cropped.iloc[val_idx].reset_index(drop=True)

        print(f"[FOLD] Train: {len(df_train)} patches", flush=True)
        print(f"[FOLD] Val:   {len(df_val)} patches", flush=True)

        print(f"\n[FOLD] === Extracción de features TRAIN ===", flush=True)
        train_features, train_labels = extract_features_from_dataframe(
            df_train, encoder, device, max_patches=max_patches_per_patient
        )

        print(f"\n[FOLD] === Extracción de features VAL ===", flush=True)
        val_features, val_labels = extract_features_from_dataframe(
            df_val, encoder, device, max_patches=max_patches_per_patient
        )

        # ===== VERIFICACIÓN ANTES DE CREAR DATASETS =====
        if len(val_features) == 0 or len(train_features) == 0:
            print(f"\n[FOLD] ⚠ ERROR CRÍTICO: No hay features válidas", flush=True)
            print(f"  Train: {len(train_features)} pacientes", flush=True)
            print(f"  Val:   {len(val_features)} pacientes", flush=True)
            print(f"[FOLD] ✗ Saltando fold {fold_idx + 1}", flush=True)

            del train_features, train_labels, val_features, val_labels
            gc.collect()
            torch.cuda.empty_cache()

            continue
        # ================================================

        print("\n[FOLD] Creando datasets...", flush=True)
        train_dataset = PrecomputedPatientDataset(train_features, train_labels)
        val_dataset = PrecomputedPatientDataset(val_features, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"[FOLD] ✓ Datasets creados correctamente", flush=True)

        print("\n[FOLD] Creando modelo de atención...", flush=True)
        attention_params = {
            'in_features': feature_dim,
            'decom_space': 128,
            'ATTENTION_BRANCHES': 1
        }
        attention_model = GatedAttention(attention_params).to(device)
        print("Creando Clasificador")
        classifier_params = {
            'in_features': feature_dim,
            'out_features': 2
        }
        classifier = NeuralNetwork(classifier_params).to(device)

        optimizer = optim.Adam(
            list(attention_model.parameters()) + list(classifier.parameters()),
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        print("[FOLD] ✓ Modelo de atención creado", flush=True)

        print(f"\n[FOLD] Iniciando entrenamiento ({epochs} épocas)...", flush=True)
        best_val_acc = 0.0

        for epoch in range(epochs):
            attention_model.train()
            classifier.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for features_batch, labels_batch in train_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)

                optimizer.zero_grad()

                batch_predictions = []
                for patient_features in features_batch:
                    patient_features = patient_features.unsqueeze(0)
                    context_vector, _ = attention_model(patient_features)
                    logits = classifier(context_vector.squeeze(0))
                    batch_predictions.append(logits)

                logits_batch = torch.stack(batch_predictions)
                loss = criterion(logits_batch, labels_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(logits_batch, 1)
                total_train += labels_batch.size(0)
                correct_train += (predicted == labels_batch).sum().item()

            train_acc = 100 * correct_train / total_train

            attention_model.eval()
            classifier.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for features_batch, labels_batch in val_loader:
                    features_batch = features_batch.to(device)
                    labels_batch = labels_batch.to(device)

                    batch_predictions = []
                    for patient_features in features_batch:
                        patient_features = patient_features.unsqueeze(0)
                        context_vector, _ = attention_model(patient_features)
                        logits = classifier(context_vector.squeeze(0))
                        batch_predictions.append(logits)

                    logits_batch = torch.stack(batch_predictions)
                    loss = criterion(logits_batch, labels_batch)

                    val_loss += loss.item()
                    _, predicted = torch.max(logits_batch, 1)
                    total_val += labels_batch.size(0)
                    correct_val += (predicted == labels_batch).sum().item()

            val_acc = 100 * correct_val / total_val

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%", flush=True)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        print(f"\n[FOLD] ✓ Fold {fold_idx+1} completado. Mejor Val Acc: {best_val_acc:.2f}%", flush=True)

        del train_features, train_labels, val_features, val_labels
        del train_dataset, val_dataset, train_loader, val_loader
        del attention_model, classifier, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*80)