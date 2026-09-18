import pandas as pd
import numpy as np
from pathlib import Path

basedir = Path("C:/Users/jaime/Desktop/unisparse")

# Load CSV
#df = pd.read_csv(basedir/"ordinal_results-128.csv")
df = pd.read_csv(basedir/"ordinal_results-MDPI-ALL.csv")
#df = pd.read_csv(basedir/"ordinal_results-192.csv")

# --- Helper: extract mean values (ignore ± std) ---
metrics = ["Acc", "MAE", "QWK", "SPT", "ECE", "TIME"]
metrics = ["Acc", "MAE", "SPT", "ECE"]

for m in metrics:
    df[m] = df[m].astype(str).str.split("±").str[0].astype(float)
df = df.round(3)
print (df)
# --- Define whether higher is better ---
higher_better = {
    "Acc": True,
    "QWK": True,
    "SPT": True,
    "MAE": False,
    "ECE": False,
    "TIME": False,
}

models_to_ignore = ["UnimodalNetsparse"]

print (df)
df = df[~df["Model"].isin(models_to_ignore)]
print (df)

# --- Rank per dataset ---
ranked_results = []

for dataset, group in df.groupby("Dataset"):
    ranking = group.copy()
    
    for metric in metrics:
        ascending = not higher_better[metric]
        ranking[f"{metric}_rank"] = ranking[metric].rank(
            ascending=ascending, method="average"
        )
    
    ranked_results.append(ranking)

ranked_df = pd.concat(ranked_results)

# --- Compute average rank across datasets ---
avg_ranks = (
    ranked_df.groupby("Model")[[f"{m}_rank" for m in metrics]]
    .mean()
    .reset_index()
)

# --- Rank models globally per metric ---
for metric in metrics:
    avg_ranks[f"{metric}_overall_rank"] = avg_ranks[f"{metric}_rank"].rank()

# --- Optional: overall score across all metrics ---
avg_ranks["mean_rank_all_metrics"] = avg_ranks[
    [f"{m}_rank" for m in metrics]
].mean(axis=1)

avg_ranks["overall_rank"] = avg_ranks["mean_rank_all_metrics"].rank()

# --- Sort for readability ---
avg_ranks = avg_ranks.sort_values("overall_rank")

# --- Outputs ---
print("\n=== Per-dataset ranks (example) ===")
print(ranked_df.head())

print("\n=== Average ranks per model ===")
print(avg_ranks.sort_values("overall_rank"))

# --- Save results ---
ranked_df.to_csv(basedir/"per_dataset_rankings.csv", index=False)
avg_ranks.to_csv(basedir/"average_model_rankings.csv", index=False)