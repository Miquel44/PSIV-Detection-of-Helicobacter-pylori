import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
# import torchvision

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
    def __init__(self, csv_file, transform=None, verify_images=True):
        """
        Args:
            csv_file (str): Ruta al archivo CSV con columnas 'codi' y 'path'
            transform (callable, optional): Transformaciones a aplicar a las imágenes
        """
        # Cargar CSV o DataFrame
        if isinstance(csv_file, str):
            self.data = pd.read_csv(csv_file).reset_index(drop=True)
        else:
            self.data = csv_file.reset_index(drop=True)

        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

        # Validar imágenes válidas durante la inicialización
        self.valid_indices = []
        invalid_count = 0

        if verify_images:
            print("Verificando imágenes... (solo se hace una vez)")
            for idx in range(len(self.data)):
                img_path = self.data.loc[idx, 'PATH']
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.valid_indices.append(idx)
                except Exception:
                    invalid_count += 1
            print(f"✓ Válidas: {len(self.valid_indices)} / {len(self.data)} "
                  f"({invalid_count} descartadas)")
        else:
            # Si no verificamos, asumimos que todas son válidas
            self.valid_indices = list(range(len(self.data)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        # índice real dentro del CSV original
        real_idx = self.valid_indices[index]

        img_path = self.data.loc[real_idx, 'PATH']
        codi     = self.data.loc[real_idx, 'CODI']  # patient_id

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"¡Error! No se pudo abrir la imagen: {img_path}")
            # Retorna tensores vacíos o maneja el error como prefieras
            return torch.zeros(3, 256, 256), codi 
        except Exception as e:
            print(f"❌ Error abriendo {img_path}: {e}")
            return torch.zeros(3, 256, 256), codi

        if self.transform:
            image = self.transform(image)

        return image, codi