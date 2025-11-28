import torch
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from PIL import Image
import torchvision.transforms as transforms

def evaluate_thresholds_on_holdout(model, df_holdout, thresholds_dict, min_errors=10):
    """
    Aplica TODOS los thresholds en thresholds_dict de forma simultánea.

    thresholds_dict: { "MethodA_k2": 0.52, "MethodB_q99": 0.47, ... }
    min_errors: nº mínimo de imágenes que deben superar threshold para marcar infectado

    Devuelve:
      - df_results: métricas por threshold
      - patient_results: lista con info por paciente si se quiere
    """

    device = next(model.parameters()).device

    # Preparamos contadores por threshold
    metrics = {
        name: {
            "TP":0, "FP":0, "TN":0, "FN":0
        }
        for name in thresholds_dict
    }

    # Para devolver información paciente a paciente si se desea
    patient_results = []

    # Transform
    transform_eval = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    model.eval()

    with torch.no_grad():

        # Procesar paciente a paciente
        for _, row in df_holdout.iterrows():
            true_label = int(row["PRESENCE"])
            folders = row["FOLDERS"].split(";")

            # recolectamos TODOS los errores del paciente
            patient_errors = []

            for folder in folders:
                for imgname in os.listdir(folder):
                    img_path = os.path.join(folder, imgname)

                    try:
                        img = Image.open(img_path).convert('RGB')
                    except:
                        continue

                    tensor_img = transform_eval(img).unsqueeze(0).to(device, non_blocking=True)
                    out = model(tensor_img)
                    err = torch.mean((tensor_img - out)**2).item()

                    patient_errors.append(err)

            # Ahora aplicamos TODOS LOS THRESHOLDS A LA VEZ
            for name, th in thresholds_dict.items():

                # cuántas imágenes superan el threshold?
                n_bad = sum( e >= th for e in patient_errors )

                predicted = 1 if n_bad >= min_errors else 0

                # actualizar métricas
                if predicted == 1 and true_label == 1:
                    metrics[name]["TP"] += 1
                elif predicted == 1 and true_label == 0:
                    metrics[name]["FP"] += 1
                elif predicted == 0 and true_label == 0:
                    metrics[name]["TN"] += 1
                elif predicted == 0 and true_label == 1:
                    metrics[name]["FN"] += 1

            # store results of this patient (optional)
            patient_results.append({
                "CODI": row["CODI"],
                "true": true_label,
                **{ f"pred_{name}": (sum(e>=t for e in patient_errors)>=min_errors)
                    for name,t in thresholds_dict.items() }
            })

    # Convertir métricas a dataframe
    rows = []
    for name, m in metrics.items():

        TP, FP, TN, FN = m["TP"], m["FP"], m["TN"], m["FN"]

        sensitivity = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        accuracy = (TP+TN) / (TP+FP+TN+FN)
        bal_acc = (sensitivity + specificity) / 2

        rows.append({
            "name": name,
            "threshold": thresholds_dict[name],
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": bal_acc
        })

    df_results = pd.DataFrame(rows)

    return df_results, patient_results
