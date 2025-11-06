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
from torch.utils.data import DataLoader
import gc
import torch


## Own Functions
from Models.AEmodels import AutoEncoderCNN


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

# 0.2 FOLDERS



#### 1. LOAD DATA: Implement 
# 1.1 Patient Diagnosis


# 1.2 Patches Data


#### 2. DATA SPLITING INTO INDEPENDENT SETS

# 2.0 Annotated set for FRed optimal threshold

# 2.1 AE trainnig set

# 2.1 Diagosis crossvalidation set

#### 3. lOAD PATCHES

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

# Free GPU Memory After Training
gc.collect()
torch.cuda.empty_cache()
#### 5. AE RED METRICS THRESHOLD LEARNING

## 5.1 AE Model Evaluation

# Free GPU Memory After Evaluation
gc.collect()
torch.cuda.empty_cache()

## 5.2 RedMetrics Threshold 

### 6. DIAGNOSIS CROSSVALIDATION
### 6.1 Load Patches 4 CrossValidation of Diagnosis

### 6.2 Diagnostic Power

