import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Tus librerías personalizadas
from Models.models import OneShotCNNNet
from Triplets.triplet_loss import TripletLoss
from Triplets.datasets import TripletDataset, classes_weight_binary

# --- CONFIGURACIÓN ---
ROOT_DIR = "../Data/Cropped"  # Ruta a tus imágenes
PATCH_CSV = "../Data/HP_WSI-CoordAllAnnotatedPatches.xlsx.csv" # Coordenadas de infectados
DIAGNOSIS_CSV = "../Data/PatientDiagnosis.csv" # Diagnóstico global
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
EMBEDDING_DIM = 128
MAX_SAMPLES = 2000  # ¡OJO! Limita esto si te quedas sin RAM

def load_data_for_triplet(root_dir, patch_csv, diagnosis_csv, max_samples=None):
    print("Cargando datos en memoria (esto puede tardar)...")
    
    # 1. Cargar etiquetas
    df_patches = pd.read_csv(patch_csv)
    df_diagnosis = pd.read_csv(diagnosis_csv)
    
    images_list = []
    labels_list = []
    
    # Estrategia simplificada: 
    # Label 1: Parches que están en el CSV de coordenadas (Infectados)
    # Label 0: Parches de pacientes diagnosticados como SANOS (0)
    
    # A) Cargar Positivos (Infectados)
    count = 0
    for _, row in df_patches.iterrows():
        case_id = str(row['Case_ID'])
        fname = f"{case_id}_r{row['Coord_Row']}_c{row['Coord_Col']}.png"
        path = os.path.join(root_dir, case_id, fname)
        
        if os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB').resize((64, 64)) # Resize para ahorrar memoria
                img_np = np.array(img).transpose(2, 0, 1) / 255.0 # CHW format
                images_list.append(img_np)
                labels_list.append(1)
                count += 1
            except: pass
        if max_samples and count >= max_samples // 2: break

    # B) Cargar Negativos (Sanos)
    healthy_cases = df_diagnosis[df_diagnosis['BinaryDiagnosis'] == 0]['CODI'].astype(str).tolist()
    
    count_neg = 0
    for case_id in healthy_cases:
        case_dir = os.path.join(root_dir, case_id)
        if os.path.exists(case_dir):
            patches = [p for p in os.listdir(case_dir) if p.endswith('.png')]
            for p in patches:
                try:
                    path = os.path.join(case_dir, p)
                    img = Image.open(path).convert('RGB').resize((64, 64))
                    img_np = np.array(img).transpose(2, 0, 1) / 255.0
                    images_list.append(img_np)
                    labels_list.append(0)
                    count_neg += 1
                except: pass
                if max_samples and count_neg >= max_samples // 2: break
        if max_samples and count_neg >= max_samples // 2: break
        
    return np.array(images_list), np.array(labels_list)

# --- MAIN PHASE 1 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Preparar Datos
    X, y = load_data_for_triplet(ROOT_DIR, PATCH_CSV, DIAGNOSIS_CSV, MAX_SAMPLES)
    print(f"Datos cargados: {X.shape}. Clases: {np.unique(y, return_counts=True)}")
    
    # Usamos TU datasets.py
    dataset = TripletDataset(X, y, transform=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Definir Modelo (OneShotCNNNet)
    # Simulamos los parámetros que pide tu models.py
    input_params = {'num_input_channels': 3}
    net_params = {
        'dim': 2, 
        'drop_rate': 0.3, 
        'block_configs': [[32, 32], [64, 64], [128, 128]] # Arquitectura CNN
    }
    out_params = {'prct': [1]} # Dummy param
    
    model = OneShotCNNNet(input_params, net_params, out_params).to(device)
    
    # 3. Loss y Optimizer
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 4. Bucle de Entrenamiento
    print("Iniciando entrenamiento Phase 1 (Contrastive)...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for anchor, positive, negative, _ in tqdm(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            # Extracción de características
            emb_a = model.feature_extraction(anchor)
            emb_p = model.feature_extraction(positive)
            emb_n = model.feature_extraction(negative)
            
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Triplet Loss: {total_loss/len(dataloader):.4f}")
        
    # 5. Guardar modelo
    torch.save(model.state_dict(), "backbone_phase1.pth")
    print("Modelo Fase 1 guardado.")