echo "Concatenating k-sweep results..."
python - <<'PYEOF'
import glob, os, pandas as pd
out = "results/k_sweep"
files = sorted(glob.glob(os.path.join(out, "ksweep_*.csv")))
files = [f for f in files if not f.endswith("ksweep_ALL.csv")]
if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    path = os.path.join(out, "ksweep_ALL.csv")
    df.to_csv(path, index=False)
    print(f"  Wrote {path} ({len(df)} rows)")
    # Quick sign-persistence view for linear/projection layers
    mask = df["layer_type"].str.contains("linear|attention|conv2d", case=False, regex=True)
    piv = df[mask].pivot_table(index=["model","task","layer_type"],
                               columns="k", values="rho_disp_sens")
    print("\nrho(displacement, sensitivity) across k "
          "(linear/projection/conv2d layers):")
    print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))
else:
    print("  No k-sweep files found.")
PYEOF

echo "All k-sweep runs complete."
