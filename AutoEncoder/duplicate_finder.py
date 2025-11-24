import os
import cv2
import csv
import hashlib

def image_hash(path):
    """Devuelve un hash sha256 de la imagen tras cargarla en matriz."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img_bytes = img.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()


def get_patient_folders(path):
    """Obtiene la lista de subcarpetas (pacientes) en un directorio."""
    return [d for d in os.listdir(path) 
            if os.path.isdir(os.path.join(path, d))]


def find_duplicates(cropped_dir, annotated_dir, output_csv="duplicates.csv"):
    """
    Identifica imágenes repetidas entre CROPPED y ANNOTATED y las guarda en un CSV.

    CSV generado:
        cropped_path, annotated_path
    """

    cropped_patients = set(get_patient_folders(cropped_dir))
    annotated_patients = set(get_patient_folders(annotated_dir))

    common = cropped_patients.intersection(annotated_patients)

    duplicates = []  # (cropped_path, annotated_path)

    for patient in common:
        print(f"Procesando paciente: {patient}")

        path_c = os.path.join(cropped_dir, patient)
        path_a = os.path.join(annotated_dir, patient)

        # Hashes de annotated → {hash: path}
        annotated_hashes = {}

        for img_name in os.listdir(path_a):
            img_path = os.path.join(path_a, img_name)
            h = image_hash(img_path)
            if h:
                annotated_hashes[h] = img_path

        # Comparar con cropped
        for img_name in os.listdir(path_c):
            img_path = os.path.join(path_c, img_name)
            h = image_hash(img_path)

            if h in annotated_hashes:
                duplicates.append((img_path, annotated_hashes[h]))

    # Guardar en CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cropped_path", "annotated_path"])
        writer.writerows(duplicates)

    print(f"\nCSV generado: {output_csv}")
    print(f"Total imágenes duplicadas: {len(duplicates)}")

    return duplicates

cropped = "Data/Cropped"
annotated = "Data/Annotated"

duplicates = find_duplicates(cropped, annotated, output_csv="duplicates.csv")
