echo "Concatenating summaries..."
python - <<'PYEOF'
import glob, os, pandas as pd
out = "results/scree"
summ = sorted(glob.glob(os.path.join(out, "scree_summary_*.csv")))
if summ:
    df = pd.concat([pd.read_csv(f) for f in summ], ignore_index=True)
    path = os.path.join(out, "scree_summary_ALL.csv")
    df.to_csv(path, index=False)
    print(f"  Wrote {path} ({len(df)} rows)")
else:
    print("  No summary files found.")
PYEOF

echo "All scree runs complete."
