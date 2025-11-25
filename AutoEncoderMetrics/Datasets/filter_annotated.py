import os
import pandas as pd
import re

def extract_numeric_prefix(s):
    match = re.match(r"(\d+)", str(s))
    if match:
        return int(match.group(1))
    return None

def build_annotated_csv(meta_csv_path,annotated_root,out_csv_path):
    # 1. Cargar CSV original
    meta = pd.read_csv(meta_csv_path)
    # meta = pd.read_excel(meta_csv_path)

    # 2. Mantener solo filas con Presence en {1, -1}
    meta = meta[meta["Presence"].isin([1, -1])].copy()

    # 3. Extraer parte numérica del Window_ID
    meta["Window_ID_int"] = meta["Window_ID"].apply(extract_numeric_prefix)

    # Eliminar filas cuyos Window_ID no tienen prefijo numérico válido
    meta = meta[meta["Window_ID_int"].notnull()].copy()

    # 4. Crear lookup (Pat_ID, Window_ID_int) → Presence
    lookup = {}
    for _, row in meta.iterrows():
        key = (str(row["Pat_ID"]), int(row["Window_ID_int"]))
        lookup[key] = int(row["Presence"])

    rows = []
    unmatched = 0
    total_files = 0

    # 5. Recorrer carpeta Annotated
    for root, dirs, files in os.walk(annotated_root):
        if root == annotated_root:
            continue

        folder_name = os.path.basename(root)
        pat_prefix = folder_name.split("_")[0]  # Pat_ID

        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                continue

            total_files += 1

            stem = os.path.splitext(fname)[0]

            # Extraer parte numérica del filename (Window_ID)
            win_id_int = extract_numeric_prefix(stem)
            if win_id_int is None:
                print(f"[WARN] No numeric ID in filename '{fname}' (folder {folder_name})")
                unmatched += 1
                continue

            key = (pat_prefix, win_id_int)

            if key not in lookup:
                unmatched += 1
                continue

            presence_val = lookup[key]  # 1 o -1
            presence_out = 1 if presence_val == 1 else 0

            full_path = os.path.join(root, fname)

            rows.append({
                "CODI": pat_prefix,
                "PATH": full_path,
                "PRESENCE": presence_out
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv_path, index=False)

    print(f"Total imágenes encontradas en Annotated: {total_files}")
    print(f"Total imágenes emparejadas con CSV: {len(df_out)}")
    print(f"Imágenes sin correspondencia: {unmatched}")
    print(f"CSV reconstruido guardado en: {out_csv_path}")

    return df_out


# Ejemplo de uso:
df_annotated = build_annotated_csv(
    meta_csv_path="/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/Datasets/HP_WSI-CoordAllAnnotatedPatches.csv",
    annotated_root="/fhome/maed/HelicoDataSet/CrossValidation/Annotated",
    out_csv_path="/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/DatasetsTrain/AnnotatedTrain.csv"
)
