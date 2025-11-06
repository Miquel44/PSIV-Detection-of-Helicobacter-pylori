import os
import pandas as pd
import glob
import sys

# --- 1. CONFIGURACIÓN ---

# --- Rutas de ENTRADA (Datos) ---
CROPPED_DIR = "/export/fhome/maed/HelicoDataSet/CrossValidation/Cropped"
ANNOTATED_DIR = "/export/fhome/maed/HelicoDataSet/CrossValidation/Annotated"

# Ruta al archivo CSV que nos dice el diagnóstico de cada paciente
# (El que acabas de subir)
DIAGNOSIS_CSV_PATH = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder/PatientDiagnosis.csv"


# --- Ruta de SALIDA (Dónde guardar los CSVs) ---
# Guarda los CSVs en una carpeta 'Datasets' dentro del mismo
# directorio donde está este script (junto a AEExample_Script.py)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.path.abspath('.')

OUTPUT_DIR = os.path.join(script_dir, "Datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Buscando 'Cropped' en: {CROPPED_DIR}")
print(f"Buscando 'Annotated' en: {ANNOTATED_DIR}")
print(f"Usando tabla de diagnóstico: {DIAGNOSIS_CSV_PATH}")
print(f"Guardando CSVs en: {OUTPUT_DIR}")

# --- Verificaciones de rutas ---
if not os.path.isdir(CROPPED_DIR):
    print(f"¡Error! No se encuentra la carpeta 'Cropped' en: {CROPPED_DIR}")
    sys.exit()
if not os.path.isdir(ANNOTATED_DIR):
    print(f"¡Error! No se encuentra la carpeta 'Annotated' en: {ANNOTATED_DIR}")
    sys.exit()
if not os.path.isfile(DIAGNOSIS_CSV_PATH):
    print(f"¡Error! No se encuentra el archivo de diagnóstico en: {DIAGNOSIS_CSV_PATH}")
    sys.exit()

# --- 2. GENERAR PatientDiagnosis_AE.csv (Desde 'Cropped') ---
# Propósito: Listar TODOS los parches para el entrenamiento del AE.
# CODI = ID del Paciente (nombre de la subcarpeta, SIN SUFIJO)
print(f"\n--- Generando PatientDiagnosis_AE.csv (desde {CROPPED_DIR}) ---")
cropped_patches_data = []

patient_folders_cropped = [f for f in os.listdir(CROPPED_DIR) if os.path.isdir(os.path.join(CROPPED_DIR, f))]

if not patient_folders_cropped:
    print(f"¡Advertencia! No se encontraron carpetas de pacientes en {CROPPED_DIR}")

for patient_id_folder in patient_folders_cropped: # Renombrada para claridad
    
    # ¡¡NUEVA LÍNEA!! Cortamos el sufijo (ej: 'B22-25_0' -> 'B22-25')
    base_patient_id = "_".join(patient_id_folder.split('_')[:-1])

    if not base_patient_id: # Seguridad, por si un nombre de carpeta no tiene '_'
        print(f"  Advertencia: Omitiendo carpeta con nombre extraño en Cropped: {patient_id_folder}")
        continue
        
    patient_dir_path = os.path.join(CROPPED_DIR, patient_id_folder)
    patch_files = glob.glob(os.path.join(patient_dir_path, "**", "*.png"), recursive=True)
    
    for patch_path in patch_files:
        abs_path = os.path.abspath(patch_path).replace("\\", "/")
        # ¡¡MODIFICADO!! Usamos el base_patient_id
        cropped_patches_data.append([abs_path, base_patient_id]) 

df_cropped = pd.DataFrame(cropped_patches_data, columns=['PATH', 'CODI'])
output_path_cropped = os.path.join(OUTPUT_DIR, "PatientDiagnosis_AE.csv")
df_cropped.to_csv(output_path_cropped, index=False)
print(f"Éxito: Se guardaron {len(df_cropped)} rutas de parches en {output_path_cropped}")


# --- 3. GENERAR HP_WSI-CoordAllAnnotatedPatches_AE.csv (Desde 'Annotated') ---
# Propósito: Listar parches anotados para el umbral.
# CODI = Etiqueta (0 o 1)
print(f"\n--- Generando HP_WSI-CoordAllAnnotatedPatches_AE.csv (desde {ANNOTATED_DIR}) ---")

# 1. Cargar la tabla de diagnósticos
try:
    df_diag = pd.read_csv(DIAGNOSIS_CSV_PATH)
except Exception as e:
    print(f"Error leyendo {DIAGNOSIS_CSV_PATH}: {e}")
    sys.exit()

# 2. Definir el mapeo de etiquetas (según tu descripción)
label_map = {
    "ALTA": 1,
    "BAIXA": 1,
    "NEGATIVA": 0
}

# 3. Crear un diccionario de consulta (lookup)
try:
    df_diag['LABEL'] = df_diag['DENSITAT'].map(label_map)
    diagnosis_lookup = df_diag.set_index('CODI')['LABEL'].to_dict()
    print(f"Tabla de diagnóstico cargada. {len(diagnosis_lookup)} pacientes mapeados.")
except KeyError:
    print(f"¡Error! Las columnas 'CODI' o 'DENSITAT' no se encuentran en {DIAGNOSIS_CSV_PATH}")
    sys.exit()

# 4. Recorrer la carpeta 'Annotated' y asignar etiquetas
annotated_patches_data = []
patient_folders_annotated = [f for f in os.listdir(ANNOTATED_DIR) if os.path.isdir(os.path.join(ANNOTATED_DIR, f))]

if not patient_folders_annotated:
    print(f"¡Advertencia! No se encontraron carpetas de pacientes en {ANNOTATED_DIR}")

patches_found = 0
patients_skipped = 0

for patient_id_folder in patient_folders_annotated: # Renombrada para claridad
    
    # ¡¡NUEVA LÍNEA!! Cortamos el sufijo (ej: 'B22-25_0' -> 'B22-25')
    base_patient_id = "_".join(patient_id_folder.split('_')[:-1])

    if not base_patient_id: # Seguridad
        print(f"  Advertencia: Omitiendo carpeta con nombre extraño en Annotated: {patient_id_folder}")
        patients_skipped += 1
        continue
        
    # ¡¡MODIFICADO!! Buscamos usando el base_patient_id
    label_codi = diagnosis_lookup.get(base_patient_id) 
    
    if label_codi is None or pd.isna(label_codi):
        # Este paciente de 'Annotated' no está en 'PatientDiagnosis.csv'
        print(f"  Advertencia: Omitiendo paciente '{patient_id_folder}' (base ID: {base_patient_id}). No se encontró diagnóstico o etiqueta válida en CSV.")
        patients_skipped += 1
        continue
    
    # Si encontramos el paciente, buscar todos sus parches
    patient_dir_path = os.path.join(ANNOTATED_DIR, patient_id_folder)
    patch_files = glob.glob(os.path.join(patient_dir_path, "**", "*.png"), recursive=True)
    
    for patch_path in patch_files:
        abs_path = os.path.abspath(patch_path).replace("\\", "/")
        annotated_patches_data.append([abs_path, int(label_codi)]) 
        patches_found += 1

# 5. Crear DataFrame y guardar
df_annotated = pd.DataFrame(annotated_patches_data, columns=['PATH', 'CODI'])
output_path_annotated = os.path.join(OUTPUT_DIR, "HP_WSI-CoordAllAnnotatedPatches_AE.csv")
df_annotated.to_csv(output_path_annotated, index=False)

print(f"Éxito: Se guardaron {patches_found} rutas de parches anotados en {output_path_annotated}")
if patients_skipped > 0:
    print(f"Se omitieron {patients_skipped} carpetas de pacientes de 'Annotated' (Total carpetas: {len(patient_folders_annotated)}).")

print("\n¡Proceso completado!")