# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:14:32 2024

Example of Main Steps for the Detection of HPilory using AutoEncoders for
the detection of anomalous pathological staining

Guides:
    0. Implement 2 functions for Loading Windows and metadata:
        0.1 LoadCropped to load a list of images from the Cropped folder
            inputs: list of folders containing the images, number of images to load for each folder,
                    ExcelFile with metadata
            out: Ims: list of images
                 metadata: list/array of information for each image in Ims
                           (PatID, imfilename)
        0.1 LoadAnnotated to load a list of images from the Annotated folder
            inputs: list of folders containing the images, number of images to load for each folder,
                    ExcelFile with metadata
            out: Ims: list of images
                 metadata: list/array of information for each image in Ims
                           (PatID, imfilename,presenceHelico)
                           
    1. Split Code into train and test steps 
    2. Save trainned models and any intermediate result input of the next step
    
@authors: debora gil, pau cano
email: debora@cvc.uab.es, pcano@cvc.uab.es
Reference: https://arxiv.org/abs/2309.16053 

"""

# IO Libraries
import sys
import os
import pickle

# Standard Libraries
import numpy as np
import pandas as pd
import glob

# Torch Libraries
from torch.utils.data import DataLoader, random_split
import gc
import torch
import torch.optim as optim
import torch.nn as nn

## Own Functions
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import ImageDataset

# WandB
# import wandb


def AEConfigs(Config):
    net_paramsEnc={}
    net_paramsDec={}
    inputmodule_paramsDec={}
    if Config=='1':
        # CONFIG1
        net_paramsEnc['block_configs']=[[32,32],[64,64]]
        net_paramsEnc['stride']=[[1,2],[1,2]]
        net_paramsDec['block_configs']=[[64,32],[32,inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
        
    elif Config=='2':
        # CONFIG 2
        net_paramsEnc['block_configs']=[[32],[64],[128],[256]]
        net_paramsEnc['stride']=[[2],[2],[2],[2]]
        net_paramsDec['block_configs']=[[128],[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
        
    elif Config=='3':  
        # CONFIG3
        net_paramsEnc['block_configs']=[[32],[64],[64]]
        net_paramsEnc['stride']=[[1],[2],[2]]
        net_paramsDec['block_configs']=[[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]
    
    return net_paramsEnc,net_paramsDec,inputmodule_paramsDec


######################### 0. EXPERIMENT PARAMETERS
# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3

# 0.1 NETWORK TRAINING PARAMS
# WandB Initialization
# run = wandb.init(
#     entity="Grup02DeepProject",  # Cambia esto por tu nombre o equipo en WandB
#     project="HPilory-Autoencoder",  # Cambia el nombre del proyecto
#     config={
#         "learning_rate": 0.001,
#         "architecture": "AutoEncoderCNN",
#         "batch_size": 32,
#         "epochs": 50,
#         "Config": "1"
#     },
# )
# config = wandb.config

# 0.2 FOLDERS



#### 1. LOAD DATA: Implement 
# 1.1 Patient Diagnosis


# 1.2 Patches Data


#### 2. DATA SPLITING INTO INDEPENDENT SETS

# 2.0 Annotated set for FRed optimal threshold
print("Loading Annotated Dataset for RED Metrics Thresholding...")
annotated_dataset = ImageDataset('Datasets/HP_WSI-CoordAllAnnotatedPatches_AE.csv')
# 2.1 AE trainnig set
ae_dataset = ImageDataset('Datasets/PatientDiagnosis_AE.csv')
train_size = int(0.8 * len(ae_dataset))
val_size = len(ae_dataset) - train_size

train_dataset, val_dataset = random_split(
    ae_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 2.1 Diagosis crossvalidation set

#### 3. lOAD PATCHES
print("Creating DataLoaders...")
batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

annotated_loader = DataLoader(
    annotated_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)
### 4. AE TRAINING

# EXPERIMENTAL DESIGN:
# TRAIN ON AE PATIENTS AN AUTOENCODER, USE THE ANNOTATED PATIENTS TO SET THE
# THRESHOLD ON FRED, VALIDATE FRED FOR DIAGNOSIS ON A 10 FOLD SCHEME OF REMAINING
# CASES.

# 4.1 Data Split


###### CONFIG1
Config='1'
net_paramsEnc,net_paramsDec,inputmodule_paramsDec=AEConfigs(Config)
model=AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                     inputmodule_paramsDec, net_paramsDec)
# 4.2 Model Training
print("Training AutoEncoder Model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
best_val_loss = float('inf')
save_path = f'Models/AE_Config{Config}_best.pth'

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    # Training
    model.train()
    train_loss = 0.0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        # Debug solo en el primer batch del primer epoch
        if epoch == 0 and batch_idx == 0:
            print(f"Input shape: {images.shape}")
            print(f"Expected output shape: {images.shape}")

        optimizer.zero_grad()
        outputs = model(images)

        if epoch == 0 and batch_idx == 0:
            print(f"Actual output shape: {outputs.shape}")

        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Log losses to wandb
    # wandb.log({
    #     "epoch": epoch + 1,
    #     "train_loss": train_loss,
    #     "val_loss": val_loss
    # })

    # Guardar mejor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        # wandb.run.summary["best_val_loss"] = best_val_loss
        # wandb.save(save_path)

print("Training complete.")

# Cargar mejor modelo
model.load_state_dict(torch.load(save_path))

# Free GPU Memory After Training
gc.collect()
torch.cuda.empty_cache()
#### 5. AE RED METRICS THRESHOLD LEARNING

## 5.1 AE Model Evaluation
print("Evaluating AutoEncoder Model on Annotated Dataset...")
model.eval()
reconstruction_errors = []
codis = []

with torch.no_grad():
    for images, codi in annotated_loader:
        images = images.to(device)
        outputs = model(images)

        # Error por imagen (MSE por píxel)
        errors = torch.mean((images - outputs) ** 2, dim=[1, 2, 3])
        reconstruction_errors.extend(errors.cpu().numpy())
        codis.extend(codi)

# Guardar resultados
print("Saving reconstruction errors...")
os.makedirs("Results", exist_ok=True)
results_df = pd.DataFrame({
    'CODI': codis,
    'reconstruction_error': reconstruction_errors
})
results_path = f'Results/AE_Config{Config}_errors.csv'
results_df.to_csv(results_path, index=False)

# Log reconstruction errors summary to wandb
# wandb.log({
#     "mean_reconstruction_error": np.mean(reconstruction_errors),
#     "std_reconstruction_error": np.std(reconstruction_errors)
# })

# wandb.save(results_path)

# Free GPU Memory After Evaluation
gc.collect()
torch.cuda.empty_cache()

# Finish wandb run
# wandb.finish()

## 5.2 RedMetrics Threshold 

### 6. DIAGNOSIS CROSSVALIDATION
### 6.1 Load Patches 4 CrossValidation of Diagnosis

### 6.2 Diagnostic Power

