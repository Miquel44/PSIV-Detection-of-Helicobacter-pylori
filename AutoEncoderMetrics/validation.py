import torch
from utils import rgb_to_hsv_torch
from torch.utils.data import DataLoader
from Models.datasets import ImageDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import GroupKFold
from scipy.stats import genpareto

def reconstruction_error_hsv(images, outputs):
    images_hsv = rgb_to_hsv_torch(images)
    outputs_hsv = rgb_to_hsv_torch(outputs)
    return torch.mean((images_hsv - outputs_hsv)**2, dim=[1,2,3]).detach().cpu().numpy()

def reconstruction_error_rgb(images, outputs):    
    return torch.mean((outputs - images) ** 2, dim=[1,2,3]).detach().cpu().numpy()

def reconstruction_error_hsv_mean_max(images, outputs, lam=0.5):
    images_hsv  = rgb_to_hsv_torch(images)
    outputs_hsv = rgb_to_hsv_torch(outputs)
    per_pixel   = (outputs_hsv - images_hsv) ** 2
    per_pixel   = per_pixel.mean(dim=1)   # (B, H, W)

    mean_err = per_pixel.mean(dim=[1, 2])    # (B,)
    max_err  = per_pixel.amax(dim=[1, 2])    # (B,)

    score = mean_err + lam * max_err
    return score.detach().cpu().numpy()

def reconstruction_error_hsv_mean_max_hue(images, outputs, lam=0.5):
    """
    Error de reconstrucción usando SOLO el canal Hue.
    images, outputs: tensores (B,3,H,W) en [0,1]
    """
    images_hsv  = rgb_to_hsv_torch(images)
    outputs_hsv = rgb_to_hsv_torch(outputs)

    # Solo canal Hue (índice 0) -> (B, H, W)
    hue_img = images_hsv[:, 0, :, :]
    hue_out = outputs_hsv[:, 0, :, :]

    per_pixel = (hue_out - hue_img) ** 2      # (B, H, W)

    mean_err = per_pixel.mean(dim=[1, 2])     # (B,)
    max_err  = per_pixel.amax(dim=[1, 2])     # (B,)

    score = mean_err + lam * max_err
    return score.detach().cpu().numpy()

def compute_errors_on_annotated(model, annotated_df, batch_size=32, save_csv="Annotated_Errors.csv"):

    device = next(model.parameters()).device

    annotated_loader = DataLoader(
        ImageDataset(annotated_df, verify_images=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    all_records = []
    model.eval()

    with torch.no_grad():
        processed = 0

        for images, codis in annotated_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)

            errors = reconstruction_error_hsv_mean_max_hue(images, outputs)

            # Fetch matching rows from DF
            for err in errors:
                row = annotated_df.iloc[processed]
                all_records.append({
                    "CODI": row["CODI"],
                    "PATH": row["PATH"],
                    "PRESENCE": int(row["PRESENCE"]),  # 0 = sano, 1 = HPylori
                    "ERROR": float(err)
                })
                processed += 1

    df_errors = pd.DataFrame(all_records)
    df_errors.to_csv(save_csv, index=False)
    print(f"Saved annotated error CSV to {save_csv}")

    return df_errors

###############################################
### MÉTODO A — THRESHOLD ESTADÍSTICO μ + kσ
###############################################

def compute_threshold_statistical(df, k=3.0, save_prefix="MethodA_Statistical"):
    """
    df: DataFrame con columnas PRESENCE (0/1) y ERROR
    k: factor multiplicativo de sigma (umbral = μ + k σ)
    """

    # Extraemos errores de sanos
    errors_sanos = df[df["PRESENCE"] == 0]["ERROR"].values
    errors_conta = df[df["PRESENCE"] == 1]["ERROR"].values

    # Calcular μ + kσ
    mu = np.mean(errors_sanos)
    sigma = np.std(errors_sanos)
    threshold = mu + k * sigma

    print("\n=== MÉTODO A — Threshold Estadístico ===")
    print(f"Total sanos: {len(errors_sanos)}")
    print(f"Total contaminados: {len(errors_conta)}")
    print(f"μ (mean sano) = {mu}")
    print(f"σ (std sano)  = {sigma}")
    print(f"Threshold (μ + {k}σ) = {threshold}")

    # Evaluación parche a parche
    y_true = df["PRESENCE"].values.astype(int)
    y_pred = (df["ERROR"].values >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    print("\n--- Métricas parche a parche ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Histograma visual
    plt.figure(figsize=(8,5))
    p99 = df["ERROR"].quantile(0.99)
    df_plot = df[df["ERROR"] <= p99]

    plt.hist(df_plot[df_plot["PRESENCE"]==0]["ERROR"], bins=60, alpha=0.6, label="Sanos")
    plt.hist(df_plot[df_plot["PRESENCE"]==1]["ERROR"], bins=60, alpha=0.6, label="Contaminados")
    plt.axvline(threshold, color="red", linestyle="--", label=f"T={threshold:.5f}")
    plt.title(f"Method A — Statistical Threshold (k={k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"AutoEncoderMetrics/Thresholds/{save_prefix}_Histogram.png")
    plt.close()

    # Guardar métricas
    with open(f"AutoEncoderMetrics/Thresholds/{save_prefix}_metrics.txt", "w") as f:
        f.write(f"Method A — Statistical threshold (k={k})\n")
        f.write(f"mu = {mu}\n")
        f.write(f"sigma = {sigma}\n")
        f.write(f"Threshold = {threshold}\n\n")
        f.write(f"Accuracy = {acc}\n")
        f.write(f"Sensitivity = {sens}\n")
        f.write(f"Specificity = {spec}\n")
        f.write(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}\n")

    return threshold, acc, sens, spec

def compute_threshold_percentile(df, q=0.99, save_prefix="MethodB_q0.99"):
    """
    Método B: umbral basado en el percentil q de los errores de los parches sanos.

    df: DataFrame con columnas 'PRESENCE' (0/1) y 'ERROR'
    q: percentil en [0,1], por ejemplo 0.99, 0.995, 0.999
    save_prefix: prefijo para guardar figuras y métricas
    """

    # Extraer errores de sanos y contaminados
    errors_sanos = df[df["PRESENCE"] == 0]["ERROR"].values
    errors_conta = df[df["PRESENCE"] == 1]["ERROR"].values

    # Threshold teórico: percentil q de los sanos
    threshold = np.quantile(errors_sanos, q)

    print(f"\n=== MÉTODO B — Threshold por percentil (q={q}) ===")
    print(f"Total sanos: {len(errors_sanos)}")
    print(f"Total contaminados: {len(errors_conta)}")
    print(f"Percentil q = {q}")
    print(f"Threshold = {threshold}")

    # Evaluación parche a parche
    y_true = df["PRESENCE"].values.astype(int)
    y_score = df["ERROR"].values.astype(float)
    y_pred  = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc  = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    print("\n--- Métricas parche a parche (Método B) ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Histograma (recortando cola para visualizar mejor)
    p99_global = df["ERROR"].quantile(0.99)
    df_plot = df[df["ERROR"] <= p99_global].copy()

    plt.figure(figsize=(8,5))
    plt.hist(df_plot[df_plot["PRESENCE"]==0]["ERROR"], bins=60, alpha=0.6, label="Sanos", color="green")
    plt.hist(df_plot[df_plot["PRESENCE"]==1]["ERROR"], bins=60, alpha=0.6, label="Contaminados", color="red")
    plt.axvline(threshold, color="blue", linestyle="--", label=f"T={threshold:.5f}")
    plt.title(f"Method B — Percentile Threshold (q={q})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"AutoEncoderMetrics/Thresholds/{save_prefix}_Histogram.png")
    plt.close()

    # Guardar métricas
    with open(f"AutoEncoderMetrics/Thresholds/{save_prefix}_metrics.txt", "w") as f:
        f.write(f"Method B — Percentile threshold\n")
        f.write(f"q = {q}\n")
        f.write(f"Threshold = {threshold}\n\n")
        f.write(f"Accuracy = {acc}\n")
        f.write(f"Sensitivity = {sens}\n")
        f.write(f"Specificity = {spec}\n")
        f.write(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}\n")

    return threshold, acc, sens, spec

def compute_threshold_methodC_theoretical(
    df_errors,
    num_folds=5,
    percentile=0.99,     
    save_prefix="MethodC_Theoretical"
):
    """
    Método C Teórico:
    -----------------
    K-Fold usando SOLO SANOS.
    En cada fold: threshold basado en percentil o MAD de los errores sanos.
    Luego se combinan todos los thresholds para obtener uno final.
    """

    # ============================
    # 1. Filtrar solo SANOS
    # ============================
    df_sanos = df_errors[df_errors["PRESENCE"] == 0].reset_index(drop=True)
    errors = df_sanos["ERROR"].values
    patient_ids = df_sanos["CODI"].values

    print(f"[Método C Teórico] Total parches sanos = {len(df_sanos)}")

    gkf = GroupKFold(n_splits=num_folds)
    thresholds = []

    # ============================
    # 2. K-Fold
    # ============================
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(errors, groups=patient_ids)):

        train_errors = errors[train_idx]

        th = np.quantile(train_errors, percentile)
        method_name = f"Percentil {percentile}"

        thresholds.append(th)
        print(f"  Fold {fold_idx+1}/{num_folds}: threshold = {th}")

    # ============================
    # 3. Threshold final (robusto)
    # ============================
    thresholds = np.array(thresholds)
    final_th = np.median(thresholds)

    print("\n======= RESULTADOS MÉTODO C (TEÓRICO) =======")
    print(f"Thresholds individuales: {thresholds}")
    print(f"Threshold final (mediana): {final_th}")

    # ============================
    # 4. Guardar resultados
    # ============================
    with open(f"AutoEncoderMetrics/Thresholds/{save_prefix}_thresholds.txt", "w") as f:
        f.write(f"Método C Teórico\n")
        f.write(f"Método interno: {method_name}\n")
        f.write(f"Thresholds por fold: {thresholds.tolist()}\n")
        f.write(f"Threshold final: {final_th}\n")

    return final_th, thresholds

def compute_threshold_methodC_empirical(
    df_errors,
    num_folds=5,
    save_prefix="MethodC_Empirical"
):
    """
    MÉTODO C — EMPÍRICO
    --------------------
    K-Fold agrupado por paciente usando SANOS + CONTAMINADOS.
    En cada fold se obtiene un threshold óptimo basado en Youden (ROC).
    El threshold final es la media o mediana de todos los thresholds por fold.
    """

    y = df_errors["PRESENCE"].values.astype(int)
    scores = df_errors["ERROR"].values.astype(float)
    patients = df_errors["CODI"].values

    gkf = GroupKFold(n_splits=num_folds)
    fold_thresholds = []

    print(f"\n[Método C Empírico] Total patches = {len(df_errors)}")
    print(f"Sanos = {(y==0).sum()}, Contaminados = {(y==1).sum()}")

    # ============================
    #  K-Fold ROC Youden
    # ============================
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(scores, y, groups=patients)):
        
        y_val = y[val_idx]
        scores_val = scores[val_idx]

        fpr, tpr, thresholds = roc_curve(y_val, scores_val)

        # Youden J
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_th = thresholds[best_idx]

        fold_thresholds.append(best_th)

        print(f"  Fold {fold_idx+1}/{num_folds}: threshold = {best_th:.6f}")

        # Optional: ROC plot for each fold
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0,1], [0,1], linestyle="--", color="gray")
        plt.scatter(fpr[best_idx], tpr[best_idx], color="red", label=f"T={best_th:.5f}")
        plt.title(f"ROC Fold {fold_idx+1}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"AutoEncoderMetrics/Thresholds/{save_prefix}_Fold{fold_idx+1}.png")
        plt.close()

    fold_thresholds = np.array(fold_thresholds)
    final_threshold = np.mean(fold_thresholds)     # media (se puede usar mediana)
    final_threshold_median = np.median(fold_thresholds)

    print("\n====== RESULTADOS MÉTODO C EMPÍRICO ======")
    print("Thresholds por fold:", fold_thresholds)
    print(f"Threshold final (media): {final_threshold:.6f}")
    print(f"Threshold final (mediana): {final_threshold_median:.6f}")

    # guardar resultados
    with open(f"AutoEncoderMetrics/Thresholds/{save_prefix}_thresholds.txt", "w") as f:
        f.write("Método C Empírico (ROC Youden)\n")
        f.write("Thresholds fold: " + str(fold_thresholds.tolist()) + "\n")
        f.write("Threshold media: " + str(final_threshold) + "\n")
        f.write("Threshold mediana: " + str(final_threshold_median) + "\n")

    return final_threshold, final_threshold_median, fold_thresholds

def compute_threshold_methodD_evt(
    df_errors,
    initial_percentile=0.95,   # u: percentil inicial
    target_fpr=1e-3,           # ε: false positive rate global deseado para sanos
    save_prefix="MethodD_EVT"
):
    errors_sanos = df_errors[df_errors["PRESENCE"] == 0]["ERROR"].values
    errors_sanos = np.sort(errors_sanos)
    N = len(errors_sanos)

    print(f"[Method D] Total parches sanos = {N}")

    # 1) Umbral inicial u (cola)
    u = np.quantile(errors_sanos, initial_percentile)
    tail = errors_sanos[errors_sanos > u] - u
    N_u = len(tail)
    tau = N_u / N   # fracción de sanos en la cola

    print(f"u (percentil {initial_percentile}) = {u:.6f}")
    print(f"Tamaño de la cola = {N_u} (tau = {tau:.4f})")

    if N_u < 50:
        print("⚠ WARNING: pocos puntos en la cola, EVT puede ser inestable")

    # 2) Ajuste GPD
    shape, loc, scale = genpareto.fit(tail, floc=0)
    print(f"GPD: shape ξ = {shape:.4f}, scale β = {scale:.4f}")

    # 3) Queremos P(X > T) = target_fpr
    #    => P(X > T | X > u) = target_fpr / tau
    if tau <= target_fpr:
        # Si la cola ya es más pequeña que el FPR deseado,
        # podemos usar simplemente el max de la cola como threshold.
        print("tau <= target_fpr, usando max de sanos como threshold aproximado.")
        T = errors_sanos.max()
    else:
        p_cond = 1.0 - target_fpr / tau   # prob. de estar POR DEBAJO condicional
        p_cond = np.clip(p_cond, 1e-6, 1-1e-6)

        if shape != 0:
            T = u + (scale / shape) * ((1 - p_cond)**(-shape) - 1)
        else:
            T = u + scale * np.log(1/(1 - p_cond))

    print(f"Threshold EVT con target_fpr={target_fpr}: T = {T:.6f}")

    # 4) Plot
    plt.figure(figsize=(8,5))
    plt.hist(errors_sanos, bins=80, alpha=0.6, label="Sanos")
    plt.axvline(u, color="blue", linestyle="--", label=f"u = {u:.6f}")
    plt.axvline(T, color="red", linestyle="--", label=f"EVT T = {T:.6f}")
    plt.legend()
    plt.title("EVT Threshold (Method D corregido)")
    plt.tight_layout()
    plt.savefig(f"AutoEncoderMetrics/Thresholds/{save_prefix}.png")
    plt.close()

    with open(f"AutoEncoderMetrics/Thresholds/{save_prefix}_threshold.txt", "w") as f:
        f.write("Method D – EVT (corregido)\n")
        f.write(f"u = {u}\n")
        f.write(f"N = {N}, N_u = {N_u}, tau = {tau}\n")
        f.write(f"shape e = {shape}, scale b = {scale}\n")
        f.write(f"target_fpr = {target_fpr}\n")
        f.write(f"Threshold T = {T}\n")

    return T, u, shape, scale

def compute_threshold_youden(df):
    print()
    
    # Convert to numpy arrays
    y_true = df["PRESENCE"].values.astype(int)
    y_score = df["ERROR"].values.astype(float)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC = {roc_auc:.4f}")

    # Optional: Plot ROC
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Autoencoder Reconstruction Error)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"AutoEncoderMetrics/Thresholds/ROC_Curve.png")
    plt.close()

    # Youden's J statistic = sensitivity + specificity - 1
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]

    print(f"Best threshold by Youden J = {best_threshold}")

    # Save threshold
    with open(f"best_threshold_MSELoss.txt", "w") as f:
        f.write(str(best_threshold))

    # Evaluate patch-level performance at this threshold
    y_pred = (y_score >= best_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / np.sum(cm)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\nPatch-level evaluation:")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # Save evaluation
    with open(f"AutoEncoderMetrics/Thresholds/patch_level_metrics_MSELoss.txt", "w") as f:
        f.write(f"Threshold: {best_threshold}\n")
        f.write(f"AUC: {roc_auc}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Sensitivity: {sensitivity}\n")
        f.write(f"Specificity: {specificity}\n")
        f.write(f"TN: {tn}\nFP: {fp}\nFN: {fn}\nTP: {tp}\n")

    p99 = df["ERROR"].quantile(0.99)
    df_plot = df[df["ERROR"] <= p99].copy()

    # Plot histogram of errors (sanos vs contaminados)
    plt.figure(figsize=(8,5))
    plt.hist(df_plot[df_plot["PRESENCE"]==0]["ERROR"], bins=50, alpha=0.6, label="Sanos", color="green")
    plt.hist(df_plot[df_plot["PRESENCE"]==1]["ERROR"], bins=50, alpha=0.6, label="Contaminados", color="red")
    plt.axvline(best_threshold, color="blue", linestyle="--", label=f"Threshold={best_threshold:.5f}")
    plt.title("Histogram of Reconstruction Errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"AutoEncoderMetrics/Thresholds/Error_Histogram_MSELoss_clipped.png")
    plt.close()

    return best_threshold