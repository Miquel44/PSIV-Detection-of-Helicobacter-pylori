import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PatientLatentDataset(Dataset):
    """
    Dataset a nivel paciente usando:
    - latent_features.npy / latent_labels.npy (parche)
    - AnnotatedTrain.csv para mapear CODI de cada patch
    - PatientDiagnosis.csv para la etiqueta real de paciente (DENSITAT)
    """
    def __init__(self, feats_path, labels_path,
                 annotated_csv_path, patient_csv_path,
                 task="binary"):
        self.X = np.load(feats_path).astype(np.float32)      # (N_patches, D)
        self.y_patch = np.load(labels_path).astype(np.int64) # (N_patches,)

        df_patches = pd.read_csv(annotated_csv_path)
        assert len(df_patches) == self.X.shape[0], "CSV y latent_features desincronizados"

        # CODI de cada patch
        self.codis = df_patches["CODI"].values

        # Cargar diagnóstico real de paciente
        df_pat = pd.read_csv(patient_csv_path)   # columnas: CODI, DENSITAT

        # Mapa de CODI -> label paciente
        if task == "binary":
            # NEGATIVA -> 0; BAIXA o ALTA -> 1
            def dens_to_label(d):
                return 0 if d == "NEGATIVA" else 1
            df_pat["label"] = df_pat["DENSITAT"].apply(dens_to_label)
            self.n_classes = 2
        elif task == "3class":
            mapping = {"NEGATIVA": 0, "BAIXA": 1, "ALTA": 2}
            df_pat["label"] = df_pat["DENSITAT"].map(mapping)
            self.n_classes = 3
        else:
            raise ValueError("task must be 'binary' or '3class'")

        self.patient_label_dict = dict(zip(df_pat["CODI"], df_pat["label"]))

        # Agrupar índices de parches por CODI, y asignar etiqueta paciente real
        self.patients = []
        for codi in np.unique(self.codis):
            idxs = np.where(self.codis == codi)[0]
            if codi not in self.patient_label_dict:
                # paciente que no esté en PatientDiagnosis.csv, lo ignoramos
                continue
            label_patient = int(self.patient_label_dict[codi])
            self.patients.append((codi, idxs, label_patient))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        codi, patch_idxs, label_patient = self.patients[idx]
        feats = torch.from_numpy(self.X[patch_idxs])          # (N_patches_i, D)
        y = torch.tensor(label_patient, dtype=torch.long)
        return feats, y, codi