import os
import pandas as pd
import re

def build_annotated_csv(meta_csv_path,cropped_root,out_csv_path):
    # 1. Cargar CSV original
    meta = pd.read_csv(meta_csv_path)

    # 2.1 Mantener solo filas con NEGATIVA
    meta = meta[meta["DENSITAT"] == "NEGATIVA"].copy()

    rows = []
    total_files = 0

    # 2.2 Gurdar todos los CODI de pacientes con NEGATIVA
    negative_patients = set(meta["CODI"].astype(str).tolist())

    # 3. Recorrer carpeta Cropped y si el paciente esta en meta guardar sus fotos en csv con cols=CODI,PATH
    for root, dirs, files in os.walk(cropped_root):
        if root == cropped_root:
            continue

        folder_name = os.path.basename(root)
        pat_prefix = folder_name.split("_")[0]

        if pat_prefix not in negative_patients:
            continue

        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                continue

            total_files += 1
            full_path = os.path.join(root, fname)
            rows.append({
                "CODI": pat_prefix,
                "PATH": full_path,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv_path, index=False)

    print(f"Total imágenes encontradas en Cropped: {total_files}")
    print(f"Total imágenes emparejadas con CSV: {len(df_out)}")
    print(f"CSV reconstruido guardado en: {out_csv_path}")

    return df_out


# Ejemplo de uso:
df_annotated = build_annotated_csv(
    meta_csv_path="/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/Datasets/PatientDiagnosis.csv",
    cropped_root="/fhome/maed/HelicoDataSet/CrossValidation/Cropped",
    out_csv_path="/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/DatasetsTrain/CroppedTrain.csv"
)
