import pandas as pd
import glob
import os
import sys

batch_id = sys.argv[1] if len(sys.argv) > 1 else "1"
folder_path = os.path.join(".", "results", f"batch_{batch_id}")

csv_files = glob.glob(os.path.join(folder_path, "output_ID_*.csv"))

rows = []
for file in csv_files:
    df = pd.read_csv(file)
    df = df.drop(columns=["Stderr"], errors="ignore")
    df_wide = df.pivot(index="ID", columns="Parameter", values="Value").reset_index()
    rows.append(df_wide)

combined_df = pd.concat(rows, ignore_index=True)
outpath = os.path.join("results", f"batch_{batch_id}_combined.csv")
combined_df.to_csv(outpath, index=False)

# Remove the individual CSVs to clean up
#for file in csv_files:
    #os.remove(file)

print(f"Combined CSV created: {outpath}")
