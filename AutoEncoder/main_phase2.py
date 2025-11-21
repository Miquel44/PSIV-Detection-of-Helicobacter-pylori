import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from PIL import Image
from Models.models import OneShotCNNNet
from Attention.AttentionUnits import GatedAttention

# --- CLASE WRAPPER PARA ATENCIÓN ---
# Necesitamos esto porque GatedAttention solo devuelve vectores, 
# necesitamos una capa final que diga "Sano" o "Enfermo"
class PatientDiagnoser(nn.Module):
    def __init__(self, input_dim, attention_net):
        super().__init__()
        self.attention = attention_net
        # Clasificador final: toma el contexto vector Z y predice 0 o 1
        self.classifier = nn.Linear(input_dim, 1) 
        
    def forward(self, x):
        # x: Bag of patches [1, Num_Patches, Features]
        Z, A = self.attention(x) # Z es el vector de contexto
        logits = self.classifier(Z)
        return logits, A

# --- GENERADOR DE BAGS (BOLSAS) ---
def get_patient_bag(model, patient_id, root_dir, device):
    """Extrae features de todos los parches de un paciente"""
    patient_dir = os.path.join(root_dir, str(patient_id))
    if not os.path.exists(patient_dir): return None
    
    patches = [p for p in os.listdir(patient_dir) if p.endswith('.png')]
    if len(patches) == 0: return None
    
    bag_features = []
    model.eval()
    with torch.no_grad():
        for p in patches:
            path = os.path.join(patient_dir, p)
            try:
                img = Image.open(path).convert('RGB').resize((64, 64))
                img_tensor = torch.from_numpy(np.array(img).transpose(2,0,1)/255.0).float().unsqueeze(0).to(device)
                
                # Extraemos feature y despreocupamos del gradiente
                feature = model.feature_extraction(img_tensor)
                bag_features.append(feature.cpu())
            except: pass
            
    if len(bag_features) == 0: return None
    # Retorna tensor shape [1, Num_Patches, Feature_Dim]
    return torch.cat(bag_features, dim=0).unsqueeze(0)

# --- MAIN PHASE 2 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_DIR = "./Cropped"
    
    # 1. Cargar Backbone entrenado (Congelado)
    input_params = {'num_input_channels': 3}
    net_params = {'dim': 2, 'drop_rate': 0, 'block_configs': [[32, 32], [64, 64], [128, 128]]}
    out_params = {'prct': [1]}
    
    backbone = OneShotCNNNet(input_params, net_params, out_params).to(device)
    backbone.load_state_dict(torch.load("backbone_phase1.pth"))
    backbone.eval() # Modo evaluación siempre
    
    # Averiguar dimensión de salida del backbone (haciendo pase dummy)
    dummy_in = torch.randn(1, 3, 64, 64).to(device)
    feat_dim = backbone.feature_extraction(dummy_in).shape[1]
    print(f"Dimensión de los features: {feat_dim}")

    # 2. Definir Modelo de Atención
    att_params = {
        'in_features': feat_dim, 
        'decom_space': 64,     # Espacio intermedio L
        'ATTENTION_BRANCHES': 1 
    }
    att_unit = GatedAttention(att_params) # Usamos GatedAttention del archivo subido
    mil_model = PatientDiagnoser(feat_dim, att_unit).to(device)
    
    # 3. Loop de Entrenamiento (Paciente a Paciente)
    optimizer = optim.Adam(mil_model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy
    
    df = pd.read_csv("PatientDiagnosis.csv")
    
    print("Iniciando entrenamiento Phase 2 (MIL)...")
    mil_model.train()
    
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total_patients = 0
        
        # Iteramos por pacientes (simulando un dataloader manual por simplicidad)
        for _, row in df.iterrows():
            pid = row['CODI']
            label = float(row['BinaryDiagnosis'])
            label_tensor = torch.tensor([[label]]).to(device)
            
            # A. Generar Bolsa (Bag)
            bag = get_patient_bag(backbone, pid, ROOT_DIR, device)
            if bag is None: continue
            
            bag = bag.to(device) # [1, N_patches, D]
            
            # B. Forward & Backward
            optimizer.zero_grad()
            logits, _ = mil_model(bag) # logits shape [1, 1]
            
            loss = criterion(logits, label_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calcular accuracy
            pred = torch.sigmoid(logits) > 0.5
            if pred.item() == label: correct += 1
            total_patients += 1
            
        print(f"Epoch {epoch+1} - Loss: {total_loss/total_patients:.4f} - Acc: {correct/total_patients:.2%}")