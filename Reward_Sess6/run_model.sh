#!/bin/bash
#SBATCH --output=logs/param_%A_%a.out
#SBATCH --error=logs/param_%A_%a.err
#SBATCH --array=1-5000%500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00

source ~/myenv/bin/activate

# Create variables
batch_id=$1
OFFSET=(batch_id-1)*5000 #Hardcoded
VAL=$((SLURM_ARRAY_TASK_ID + OFFSET))

# Run the Python script with the integer parameter
echo "Running with PARAM_VAL=$VAL"
PARAM_VAL=$VAL BATCH_ID=$batch_id python model_run_loco_Rew6.py