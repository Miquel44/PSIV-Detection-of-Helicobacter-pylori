#!/bin/bash
#SBATCH -n 1                    # Request 1 CPU core
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -D /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Sistema2/Joblist  # Working directory
#SBATCH -t 4-00:05              # Runtime in D-HH:MM (4 days)
#SBATCH -p tfg                  # Partition to submit to
#SBATCH --mem 16384             # Request 16 GB of RAM memory
#SBATCH -o %x_%u_%j.out         # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err         # File to which STDERR will be written
#SBATCH --gres gpu:1            # Request 1 GPU
#SBATCH --cpus-per-task=4       # Request 4 CPU cores for each task

echo "========================================"
echo "SISTEMA 2: Attention-based Patient Diagnosis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "========================================"



# Activar entorno virtual
source /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/MyVirtualEnv/bin/activate

# Información del entorno
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Ejecutar el Sistema 2
echo "Iniciando entrenamiento del Sistema 2..."
srun python /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/Sistema2/AEExample_Sistem2.py

echo ""
echo "========================================"
echo "Sistema 2 completado"
echo "Finished at: $(date)"
echo "========================================"