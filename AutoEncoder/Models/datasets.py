import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
#import torchvision

import numpy as np
from random import shuffle
import random
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

    
# =============================================================================
# Standard dataset (Single Objective)
# =============================================================================
# X needs to have structure [NSamp,...] or be a list of NSamp entries
class Standard_Dataset(data.Dataset):
    def __init__(self, X, Y=None, transformation=None):
        super().__init__()
        self.X = X
        self.y = Y
        self.transformation = transformation
 
    def __len__(self):
        
        return len(self.X)

    def __getitem__(self, idx):
        
        if self.y is not None:
            return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(np.array(self.y[idx]))
        else:
            return torch.from_numpy(self.X[idx])

             

class ImageDataset(data.Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Ruta al archivo CSV con columnas 'codi' y 'path'
            transform (callable, optional): Transformaciones a aplicar a las imágenes
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['PATH']
        img_path = img_path.replace("Data\\Cropped", "/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped")
        img_path = img_path.replace("\\", "/")
        codi = self.data.iloc[idx]['CODI']

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, codi





