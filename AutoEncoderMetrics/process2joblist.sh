#!/bin/bash
#SBATCH -n 1 # Request 4 CPU �s cores . Maximum 10 CPU �s cores .
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/Joblist # Working directory
#SBATCH -t 4-00:05 # Runtime in D-HH:MM
#SBATCH -p tfg # Partition to submit to
#SBATCH --mem 12288 # Request 12 GB of RAM memory . Maximum 60 GB.
#SBATCH -o MSElossTRAIN_%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e MSElossTRAIN_%x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 # Request 1 GPU . Maximum 8 GPUs
#SBATCH --cpus-per-task=4 # Request 4 CPU �s cores for each task

sleep 3
source /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/MyVirtualEnv/bin/activate
srun python /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoderMetrics/AEExample_Script.py

# srun python /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder/train_phase1_backbone.py
# srun python /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder/train_phase2_patch_classifier.py
# srun python /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder/precompute_embeddings.py
# srun python /export/fhome/maed01/PSIV-Detection-of-Helicobacter-pylori/AutoEncoder/train_phase3_patient_diagnosis.py