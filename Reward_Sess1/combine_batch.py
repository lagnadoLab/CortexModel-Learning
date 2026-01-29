import pandas as pd
import glob
import os
import sys

batch_id = sys.argv[1] if len(sys.argv) > 1 else "1"
folder_path = os.path.join(".", "results")

csv_files = glob.glob(os.path.join(folder_path, "batch_*.csv"))

dfs = [pd.read_csv(f) for f in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

outpath = os.path.join("results", f"combined_results_RS1_1Step.csv")
combined_df.to_csv(outpath, index=False)

# Remove the individual CSVs to clean up
#for file in csv_files:
    #os.remove(file)

print(f"Combined CSV created: {outpath}")
