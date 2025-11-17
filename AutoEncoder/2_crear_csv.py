import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
import shutil
from typing import Tuple, Dict, List
import json


class QuironDatasetOrganizer:
    """
    Organiza el dataset Quiron para entrenamiento de AutoEncoder con detección de anomalías.
    Estrategia: Entrenar solo con patches sanos (negativos) y detectar patches con H. pylori
    como anomalías (alto error de reconstrucción).
    """
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: Ruta base donde está el dataset Quiron
        """
        self.base_path = Path(base_path)
        self.cv_path = self.base_path / "CrossValidation"
        self.holdout_path = self.base_path / "HoldOut"
        
        # Archivos de metadata
        self.all_annotations_file   =       "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Datasets/HP_WSI-CoordAllAnnotatedPatches.xlsx"
        self.patient_diagnosis_file =       "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Datasets/PatientDiagnosis.csv"
        
        # DataFrames
        self.annotations_df = None
        self.patient_diagnosis_df = None
        
    def load_metadata(self):
        """Carga los archivos de metadata"""
        print("Cargando metadata...")
        
        # Cargar anotaciones (usar el archivo con TODAS las anotaciones)
        self.annotations_df = pd.read_excel(self.all_annotations_file)
        print(f"  - Anotaciones cargadas: {len(self.annotations_df)} patches")
        
        # Cargar diagnóstico de pacientes
        self.patient_diagnosis_df = pd.read_csv(self.patient_diagnosis_file)
        print(f"  - Diagnósticos de pacientes cargados: {len(self.patient_diagnosis_df)} pacientes")
        
        # Estadísticas
        n_positive = len(self.annotations_df[self.annotations_df['Presence'] == 1])
        n_negative = len(self.annotations_df[self.annotations_df['Presence'] == -1])
        n_uncertain = len(self.annotations_df[self.annotations_df['Presence'] == 0])
        
        print(f"\nEstadísticas de patches anotados:")
        print(f"  - Positivos (H. pylori presente): {n_positive}")
        print(f"  - Negativos (sanos): {n_negative}")
        print(f"  - Inciertos: {n_uncertain}")
        
        return self.annotations_df, self.patient_diagnosis_df
    
    def filter_augmented_patches(self, df: pd.DataFrame, include_augmented: bool = False) -> pd.DataFrame:
        """
        Filtra patches aumentados (con tag _Aug)
        
        Args:
            df: DataFrame con las anotaciones
            include_augmented: Si True, incluye los patches aumentados
        """
        if include_augmented:
            return df
        else:
            # Filtrar patches que NO contengan '_Aug' en Window_ID
            mask = ~df['Window_ID'].astype(str).str.contains('_Aug', na=False)
            return df[mask]
    
    def prepare_patient_level_split(self, 
                                    test_size: float = 0.2,
                                    val_size: float = 0.15,
                                    random_state: int = 42,
                                    include_augmented: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Divide los datos a nivel de PACIENTE (no de patch) para evitar data leakage.
        
        Args:
            test_size: Proporción de pacientes para test
            val_size: Proporción de pacientes para validación (del total)
            random_state: Semilla aleatoria
            include_augmented: Si incluir patches aumentados
            
        Returns:
            Diccionario con DataFrames de train, val y test
        """
        print("\n=== PREPARANDO DIVISIÓN A NIVEL DE PACIENTE ===")
        
        if self.annotations_df is None:
            self.load_metadata()
        
        # Filtrar patches aumentados si es necesario
        df = self.filter_augmented_patches(self.annotations_df, include_augmented)
        print(f"Trabajando con {len(df)} patches (augmented={'incluidos' if include_augmented else 'excluidos'})")
        
        # Obtener pacientes únicos del CrossValidation
        unique_patients = df['Pat_ID'].unique()
        print(f"Pacientes únicos en CrossValidation: {len(unique_patients)}")
        
        # Obtener etiquetas de diagnóstico para cada paciente
        patient_diagnosis_dict = dict(zip(
            self.patient_diagnosis_df['CODI'],
            self.patient_diagnosis_df.iloc[:, 1]  # Segunda columna con diagnóstico
        ))
        
        # Crear etiquetas binarias (0=NEGATIVA, 1=POSITIVA)
        patient_labels = []
        valid_patients = []
        for patient in unique_patients:
            if patient in patient_diagnosis_dict:
                diagnosis = patient_diagnosis_dict[patient]
                label = 0 if diagnosis == 'NEGATIVA' else 1
                patient_labels.append(label)
                valid_patients.append(patient)
        
        patient_labels = np.array(patient_labels)
        valid_patients = np.array(valid_patients)
        
        print(f"\nDistribución de pacientes:")
        print(f"  - Negativos: {np.sum(patient_labels == 0)}")
        print(f"  - Positivos: {np.sum(patient_labels == 1)}")
        
        # Primera división: train+val vs test (estratificada)
        train_val_patients, test_patients, train_val_labels, test_labels = train_test_split(
            valid_patients,
            patient_labels,
            test_size=test_size,
            stratify=patient_labels,
            random_state=random_state
        )
        
        # Segunda división: train vs val (estratificada)
        val_ratio = val_size / (1 - test_size)  # Ajustar proporción
        train_patients, val_patients, train_labels, val_labels = train_test_split(
            train_val_patients,
            train_val_labels,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=random_state
        )
        
        print(f"\nDivisión de pacientes:")
        print(f"  Train: {len(train_patients)} pacientes (Neg: {np.sum(train_labels==0)}, Pos: {np.sum(train_labels==1)})")
        print(f"  Val:   {len(val_patients)} pacientes (Neg: {np.sum(val_labels==0)}, Pos: {np.sum(val_labels==1)})")
        print(f"  Test:  {len(test_patients)} pacientes (Neg: {np.sum(test_labels==0)}, Pos: {np.sum(test_labels==1)})")
        
        # Obtener patches de cada conjunto de pacientes
        train_patches = df[df['Pat_ID'].isin(train_patients)].copy()
        val_patches = df[df['Pat_ID'].isin(val_patients)].copy()
        test_patches = df[df['Pat_ID'].isin(test_patients)].copy()
        
        # IMPORTANTE: Para AutoEncoder, el train debe ser SOLO NEGATIVOS
        train_patches_negative = train_patches[train_patches['Presence'] == -1].copy()
        
        print(f"\nDistribución de patches:")
        print(f"  Train (SOLO NEGATIVOS): {len(train_patches_negative)} patches")
        print(f"  Val   (Neg: {len(val_patches[val_patches['Presence']==-1])}, "
              f"Pos: {len(val_patches[val_patches['Presence']==1])})")
        print(f"  Test  (Neg: {len(test_patches[test_patches['Presence']==-1])}, "
              f"Pos: {len(test_patches[test_patches['Presence']==1])})")
        
        return {
            'train': train_patches_negative,
            'val': val_patches,
            'test': test_patches,
            'train_patients': train_patients,
            'val_patients': val_patients,
            'test_patients': test_patients
        }
    
    def prepare_kfold_split(self, 
                           n_splits: int = 5,
                           random_state: int = 42,
                           include_augmented: bool = False) -> List[Dict[str, pd.DataFrame]]:
        """
        Prepara K-Fold Cross-Validation a nivel de paciente.
        
        Args:
            n_splits: Número de folds
            random_state: Semilla aleatoria
            include_augmented: Si incluir patches aumentados
            
        Returns:
            Lista de diccionarios, uno por fold, con train y val DataFrames
        """
        print(f"\n=== PREPARANDO {n_splits}-FOLD CROSS-VALIDATION ===")
        
        if self.annotations_df is None:
            self.load_metadata()
        
        # Filtrar patches aumentados si es necesario
        df = self.filter_augmented_patches(self.annotations_df, include_augmented)
        
        # Obtener pacientes únicos
        unique_patients = df['Pat_ID'].unique()
        
        # Obtener etiquetas de diagnóstico
        patient_diagnosis_dict = dict(zip(
            self.patient_diagnosis_df['CODI'],
            self.patient_diagnosis_df.iloc[:, 1]
        ))
        
        patient_labels = []
        valid_patients = []
        for patient in unique_patients:
            if patient in patient_diagnosis_dict:
                diagnosis = patient_diagnosis_dict[patient]
                label = 0 if diagnosis == 'NEGATIVA' else 1
                patient_labels.append(label)
                valid_patients.append(patient)
        
        patient_labels = np.array(patient_labels)
        valid_patients = np.array(valid_patients)
        
        # Crear K-Fold estratificado
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(valid_patients, patient_labels)):
            print(f"\nFold {fold_idx + 1}/{n_splits}:")
            
            train_patients = valid_patients[train_idx]
            val_patients = valid_patients[val_idx]
            
            train_labels = patient_labels[train_idx]
            val_labels = patient_labels[val_idx]
            
            print(f"  Train: {len(train_patients)} pacientes (Neg: {np.sum(train_labels==0)}, Pos: {np.sum(train_labels==1)})")
            print(f"  Val:   {len(val_patients)} pacientes (Neg: {np.sum(val_labels==0)}, Pos: {np.sum(val_labels==1)})")
            
            # Obtener patches
            train_patches = df[df['Pat_ID'].isin(train_patients)].copy()
            val_patches = df[df['Pat_ID'].isin(val_patients)].copy()
            
            # Train SOLO con negativos
            train_patches_negative = train_patches[train_patches['Presence'] == -1].copy()
            
            print(f"  Patches - Train (SOLO NEG): {len(train_patches_negative)}, "
                  f"Val (Neg: {len(val_patches[val_patches['Presence']==-1])}, "
                  f"Pos: {len(val_patches[val_patches['Presence']==1])})")
            
            folds.append({
                'fold': fold_idx + 1,
                'train': train_patches_negative,
                'val': val_patches,
                'train_patients': train_patients,
                'val_patients': val_patients
            })
        
        return folds
    
    def get_patch_paths(self, df: pd.DataFrame) -> List[str]:
        """
        Obtiene las rutas completas de los archivos de patches.
        
        Args:
            df: DataFrame con columnas Pat_ID, Section_ID, Window_ID
            
        Returns:
            Lista de rutas a los archivos .png
        """
        paths = []
        for _, row in df.iterrows():
            patient_folder = f"{row['Pat_ID']}_{row['Section_ID']}"
            patch_filename = f"{row['Window_ID']}.png"
            
            # Buscar en Annotated primero, luego en Cropped
            annotated_path = self.cv_path / "Annotated" / patient_folder / patch_filename
            cropped_path = self.cv_path / "Cropped" / patient_folder / patch_filename
            
            if annotated_path.exists():
                paths.append(str(annotated_path))
            elif cropped_path.exists():
                paths.append(str(cropped_path))
            else:
                paths.append(None)  # No encontrado
        
        return paths
    
    def save_split_info(self, split_data: Dict, output_dir: str):
        """
        Guarda información de la división en archivos CSV y JSON.
        
        Args:
            split_data: Diccionario retornado por prepare_patient_level_split o prepare_kfold_split
            output_dir: Directorio donde guardar los archivos
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGuardando información de la división en {output_dir}...")
        
        # Guardar DataFrames de patches
        for split_name in ['train', 'val', 'test']:
            if split_name in split_data and split_data[split_name] is not None:
                df = split_data[split_name].copy()
                # Añadir rutas de archivos
                df['patch_path'] = self.get_patch_paths(df)
                # Guardar
                output_file = output_path / f"{split_name}_patches.csv"
                df.to_csv(output_file, index=False)
                print(f"  - {split_name}_patches.csv guardado ({len(df)} patches)")
        
        # Guardar listas de pacientes
        patient_info = {}
        if 'train_patients' in split_data:
            patient_info['train_patients'] = split_data['train_patients'].tolist()
        if 'val_patients' in split_data:
            patient_info['val_patients'] = split_data['val_patients'].tolist()
        if 'test_patients' in split_data:
            patient_info['test_patients'] = split_data['test_patients'].tolist()
        
        with open(output_path / 'patient_split.json', 'w') as f:
            json.dump(patient_info, f, indent=2)
        print(f"  - patient_split.json guardado")
        
        # Resumen
        summary = {
            'total_train_patches': len(split_data['train']),
            'total_val_patches': len(split_data['val']),
            'train_patients': len(split_data['train_patients']),
            'val_patients': len(split_data['val_patients']),
            'val_positive_patches': len(split_data['val'][split_data['val']['Presence'] == 1]),
            'val_negative_patches': len(split_data['val'][split_data['val']['Presence'] == -1])
        }
        
        # Añadir info de test solo si existe
        if 'test' in split_data and split_data['test'] is not None and len(split_data['test']) > 0:
            summary['total_test_patches'] = len(split_data['test'])
            summary['test_patients'] = len(split_data['test_patients'])
            summary['test_positive_patches'] = len(split_data['test'][split_data['test']['Presence'] == 1])
            summary['test_negative_patches'] = len(split_data['test'][split_data['test']['Presence'] == -1])
        
        with open(output_path / 'split_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  - split_summary.json guardado")
        
        print("\n¡División completada y guardada exitosamente!")


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Configuración
    BASE_PATH = "/export/fhome/maed/HelicoDataSet/"  # CAMBIAR ESTA RUTA
    OUTPUT_DIR = "/export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/data_splits"
    
    # Inicializar organizador
    organizer = QuironDatasetOrganizer(BASE_PATH)
    
    # Cargar metadata
    organizer.load_metadata()
    
    # OPCIÓN 1: División simple train/val/test
    print("\n" + "="*70)
    print("OPCIÓN 1: División simple train/val/test")
    print("="*70)
    
    split_data = organizer.prepare_patient_level_split(
        test_size=0.2,
        val_size=0.15,
        random_state=42,
        include_augmented=False  # No incluir patches aumentados en train
    )
    
    # Guardar información
    organizer.save_split_info(split_data, OUTPUT_DIR)
    
    # Acceder a los datos
    print("\nEjemplo de acceso a los datos:")
    print(f"Train patches (SOLO NEGATIVOS): {len(split_data['train'])}")
    print(f"Primeros 3 patches de train:")
    print(split_data['train'].head(3))
    
    # OPCIÓN 2: K-Fold Cross-Validation
    print("\n" + "="*70)
    print("OPCIÓN 2: K-Fold Cross-Validation")
    print("="*70)
    
    folds = organizer.prepare_kfold_split(
        n_splits=5,
        random_state=42,
        include_augmented=False
    )
    
    # Guardar cada fold
    for fold_data in folds:
        fold_num = fold_data['fold']
        fold_dir = f"{OUTPUT_DIR}/fold_{fold_num}"
        
        # Crear estructura para el fold (sin test)
        fold_split = {
            'train': fold_data['train'],
            'val': fold_data['val'],
            'train_patients': fold_data['train_patients'],
            'val_patients': fold_data['val_patients']
        }
        organizer.save_split_info(fold_split, fold_dir)
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
    print(f"\nArchivos generados en: {OUTPUT_DIR}")
    print("\nPróximos pasos:")
    print("1. Cargar los CSV generados en tu código de entrenamiento")
    print("2. Usar train_patches.csv para entrenar el AutoEncoder (solo negativos)")
    print("3. Usar val_patches.csv para monitorizar y ajustar umbral de detección")
    print("4. Usar test_patches.csv para evaluación final")
    print("5. Usar los patches del HoldOut para verificar reproducibilidad")