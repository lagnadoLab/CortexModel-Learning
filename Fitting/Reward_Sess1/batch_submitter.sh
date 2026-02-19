#!/bin/bash
#SBATCH --time=150:00:00
#SBATCH --partition=sussexneuro
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=batch_controller

# ======= USER SETTINGS =======
total_batches=10    # total number of batches you want to run
max_active=4  # how many batches to keep active at once
declare -A active_jobs
current=1
suffix="_RS1"
# ==============================

submit_batch () {
    local batch=$1
    echo "=== Starting batch $batch of $total_batches ==="
    # 0. Choose partition
    partitions=("short" "general")
    best_partition=""
    min_jobs=999999

    for p in "${partitions[@]}"; do
    # Count how many of your jobs are running in this partition
        n_jobs=$(squeue -u $USER -h -o "%i %P" | awk -v p="$p" '$2==p {print $1}' | sed 's/_.*//' | sort -u | wc -l)
        echo "Currently $n_jobs jobs in $p partition."

        if [ "$n_jobs" -lt "$min_jobs" ]; then
            min_jobs=$n_jobs
            best_partition=$p
        fi
    done
	mkdir -p logs/batch_${batch} results/batch_${batch}

    # 1. Submit compute array for this batch
    jid=$(sbatch \
        --partition=${best_partition} \
	--job-name="${batch}${suffix}" \
        --output=logs/batch_${batch}/param_%A_%a.out \
        --error=logs/batch_${batch}/param_%A_%a.err \
        run_model.sh $batch | awk '{print $4}')
    echo "Submitted batch $batch with JobID $jid"

    # 2. Submit concatenation step dependent on the compute job finishing
    sbatch \
        --dependency=afterany:$jid \
	--partition=${best_partition} \
	--job-name=concat_${batch} \
      	--output=logs/batch_${batch}/concat_%A.out \
      	--error=logs/batch_${batch}/concat_%A.err \
	--wrap="python combine_results.py ${batch}" 
    }
    

# --- main control loop ---
current=1
while [ $current -le $total_batches ]; do
    # count running batches
    running=$(squeue -u "$USER" -h -o "%.30j" | grep "$suffix" | sort -u | wc -l)
	echo "running jobs = $running"
    if [ $running -lt $max_active ]; then
        submit_batch $current
        ((current++))
	sleep 30
    else
        echo "Waiting... $running batches active."
        sleep 600  # check every 10 minutes
    fi
done

echo "All batches submitted and monitored."