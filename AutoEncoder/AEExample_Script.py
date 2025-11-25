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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

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
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsDec = {}
    if Config == '1':
        # CONFIG1
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    elif Config == '2':
        # CONFIG 2
        net_paramsEnc['block_configs'] = [[32], [64], [128], [256]]
        net_paramsEnc['stride'] = [[2], [2], [2], [2]]
        net_paramsDec['block_configs'] = [[128], [64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    elif Config == '3':
        # CONFIG3
        net_paramsEnc['block_configs'] = [[32], [64], [64]]
        net_paramsEnc['stride'] = [[1], [2], [2]]
        net_paramsDec['block_configs'] = [[64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


if __name__ == "__main__":
    ######################### 0. EXPERIMENT PARAMETERS
    # 0.1 AE PARAMETERS
    inputmodule_paramsEnc = {}
    inputmodule_paramsEnc['num_input_channels'] = 3
    num_folds = 3
    batch_size = 32
    epochs = 3
    learning_rate = 1e-3
    train = True

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
    cropped = "Data/Cropped"
    annotated = "Data/Annotated"
    hold = "Data/HoldOut"

    #### 1. LOAD DATA: Implement
    # 1.1 Patient Diagnosis
    df_cropped = pd.read_csv("/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Data/PatientDiagnosis_AE_Linux.csv")
    image_paths = df_cropped["PATH"].tolist()
    patient_ids = df_cropped["CODI"].tolist()

    # 1.2 Patches Data
    # df_annotated = pd.read_csv("Data/annotated.csv")

    # 1.3 HoldOut Data
    # df_holdout = pd.read_csv("Datasets/HoldOut_clean.csv")

    #### 2. DATA SPLITING INTO INDEPENDENT SETS

    if num_folds > 1 and train:
        gkf = GroupKFold(n_splits=num_folds)

        fold_loaders = []  # almacenamos tuplas (train_loader, val_loader) por fold

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(image_paths, groups=patient_ids)):
            print(f"\n========= FOLD {fold_idx} / {num_folds} =========")

            # Subsets del CSV
            df_train_fold = df_cropped.iloc[train_idx]
            df_val_fold = df_cropped.iloc[val_idx]

            # Dataset por fold
            fold_train_dataset = ImageDataset(df_train_fold, verify_images=False)
            fold_val_dataset = ImageDataset(df_val_fold, verify_images=False)

            # DataLoaders por fold
            train_loader = DataLoader(
                fold_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )

            val_loader = DataLoader(
                fold_val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )

            fold_loaders.append((train_loader, val_loader))

            print(f"Train images: {len(fold_train_dataset)}")
            print(f"Val images:   {len(fold_val_dataset)}")

    #### 3. lOAD PATCHES
    print("Creating DataLoaders...")

    if num_folds <= 1 and train:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(image_paths, groups=patient_ids))

        df_train_final = df_cropped.iloc[train_idx].reset_index(drop=True)
        df_val_final = df_cropped.iloc[val_idx].reset_index(drop=True)

        train_final_loader = DataLoader(
            ImageDataset(df_train_final, verify_images=False),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        val_final_loader = DataLoader(
            ImageDataset(df_val_final, verify_images=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

    # annotated_loader = DataLoader(
    #     ImageDataset(df_annotated, verify_images=False),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    #     prefetch_factor=2,
    #     persistent_workers=True
    # )

    # holdout_loader = DataLoader(
    #     ImageDataset(df_holdout),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     prefetch_factor=2,
    #     persistent_workers=True
    # )

    ### 4. AE TRAINING

    # EXPERIMENTAL DESIGN:
    # TRAIN ON AE PATIENTS AN AUTOENCODER, USE THE ANNOTATED PATIENTS TO SET THE
    # THRESHOLD ON FRED, VALIDATE FRED FOR DIAGNOSIS ON A 10 FOLD SCHEME OF REMAINING
    # CASES.

    ###### CONFIG
    Config = '1'
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()


    # 4.1 Model Training
    def train_one_model(train_loader, val_loader, fold_idx=None):
        """Entrena un modelo (para un fold o para el entrenamiento clásico)."""

        model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                               inputmodule_paramsDec, net_paramsDec)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        tag = f"_fold{fold_idx}" if fold_idx is not None else ""
        save_path = f"/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder/Models/Trained/AE_Config{Config}{tag}_best_Linux.pth"

        print(f"\n>>> Training AutoEncoder (Config {Config}{tag})...")
        first_batch_debug_done = False

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            # -------- TRAIN --------
            model.train()
            train_loss = 0.0

            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)

                # Debug de shapes solo una vez
                if not first_batch_debug_done:
                    print(f"Input shape: {images.shape}")
                    first_batch_debug_done = True

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # -------- VALIDATION --------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, images)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f'Epoch {epoch + 1}/{epochs}, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)

        print(f"Training complete. Best val loss: {best_val_loss:.4f}")

        # Cargar el mejor modelo antes de devolverlo
        model.load_state_dict(torch.load(save_path, map_location=device))

        # Liberar algo de memoria
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model, best_val_loss


    # Make
    fold_val_losses = []

    if train:
        if num_folds > 1:
            # ----- K-FOLD -----
            for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
                print(f"\n########## FOLD {fold_idx + 1} / {num_folds} ##########")
                model_fold, best_val = train_one_model(train_loader, val_loader, fold_idx=fold_idx)
                fold_val_losses.append(best_val)

            print("\n===== Cross-validation results =====")
            for i, v in enumerate(fold_val_losses):
                print(f"Fold {i}: best val loss = {v:.4f}")
            print(f"Mean val loss: {np.mean(fold_val_losses):.4f} | "
                  f"Std: {np.std(fold_val_losses):.4f}")

        else:
            # ----- ENTRENAMIENTO CLÁSICO -----
            # Aquí puedes decidir si:
            #  - usas un split train/val dentro de df_cropped, o
            #  - entrenas con cropped_loader y validas con annotated_loader
            model_final, best_val = train_one_model(train_final_loader, val_final_loader, fold_idx=None)

    # Free GPU Memory After Training
    gc.collect()
    torch.cuda.empty_cache()

    if num_folds > 1 and train:
        exit()


    #### 5. AE RED METRICS THRESHOLD LEARNING
    def reconstruction_error(images, outputs):
        return torch.mean((outputs - images) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()


    def compute_errors_on_annotated(model, annotated_df, batch_size=32, save_csv="Annotated_Errors.csv"):

        device = next(model.parameters()).device

        annotated_loader = DataLoader(
            ImageDataset(annotated_df, verify_images=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        all_records = []
        model.eval()

        with torch.no_grad():
            processed = 0

            for images, codis in annotated_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)

                errors = reconstruction_error(images, outputs)

                # Fetch matching rows from DF
                for err in errors:
                    row = annotated_df.iloc[processed]
                    all_records.append({
                        "CODI": row["CODI"],
                        "PATH": row["PATH"],
                        "PRESENCE": int(row["PRESENCE"]),  # 0 = sano, 1 = HPylori
                        "ERROR": float(err)
                    })
                    processed += 1

        df_errors = pd.DataFrame(all_records)
        df_errors.to_csv(save_csv, index=False)
        print(f"Saved annotated error CSV to {save_csv}")

        return df_errors


    ## 5.1 AE Model Evaluation
    print("Evaluating AutoEncoder Model on Annotated Dataset...")

    if not train:
        model_final = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                                     inputmodule_paramsDec, net_paramsDec)

        model_final.load_state_dict(torch.load(
            "/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder/Models/Trained/AE_Config1_fold1_best_Linux.pth",
            map_location=device))
        model_final = model_final.to(device)

    df_errors_annotated = compute_errors_on_annotated(
        model_final,
        df_annotated,  # <-- debe tener CODI, PATH, PRESENCE
        batch_size=32,
        save_csv="/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder/Results/Annotated_Errors_Linux.csv"
    )

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

