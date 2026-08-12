#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    df = pd.read_csv(csv_path)

    required = {
        "processes", "total_samples", "abs_error", "total_elapsed_sec"
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in CSV: {sorted(missing)}")

    df = df.sort_values(["total_samples", "processes"]).reset_index(drop=True)

    plt.figure()
    for p, g in df.groupby("processes"):
        plt.loglog(g["total_samples"], g["abs_error"], marker="o", label=f"P={p}")
    plt.xlabel("Total Monte Carlo samples")
    plt.ylabel("Absolute error")
    plt.title("Accuracy of Monte Carlo approximation")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy.png", dpi=200)

    largest_n = df["total_samples"].max()
    gmax = df[df["total_samples"] == largest_n].copy()
    plt.figure()
    plt.plot(gmax["processes"], gmax["total_elapsed_sec"], marker="o")
    plt.xlabel("MPI processes")
    plt.ylabel("Elapsed time, seconds")
    plt.title(f"Runtime for N={largest_n}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("runtime.png", dpi=200)


    rows = []
    for n, g in df.groupby("total_samples"):
        if (g["processes"] == 1).any():
            baseline_row = g[g["processes"] == 1].iloc[0]
        else:
            baseline_row = g.iloc[g["processes"].argmin()]
        baseline_p = int(baseline_row["processes"])
        baseline_t = float(baseline_row["total_elapsed_sec"])
        for _, row in g.iterrows():
            p = int(row["processes"])
            t = float(row["total_elapsed_sec"])
            speedup = baseline_t / t if t > 0 else float("nan")
            efficiency = speedup / (p / baseline_p)
            rows.append({
                "total_samples": n,
                "processes": p,
                "baseline_processes": baseline_p,
                "speedup": speedup,
                "efficiency": efficiency,
            })

    metrics = pd.DataFrame(rows)
    out_metrics = df.merge(metrics, on=["total_samples", "processes"], how="left")
    out_metrics.to_csv("results_with_metrics.csv", index=False)

    plt.figure()
    for n, g in metrics.groupby("total_samples"):
        plt.plot(g["processes"], g["speedup"], marker="o", label=f"N={n}")
    plt.xlabel("MPI processes")
    plt.ylabel("Speedup")
    plt.title("Speedup")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("speedup.png", dpi=200)

    plt.figure()
    for n, g in metrics.groupby("total_samples"):
        plt.plot(g["processes"], g["efficiency"], marker="o", label=f"N={n}")
    plt.xlabel("MPI processes")
    plt.ylabel("Efficiency")
    plt.title("Efficiency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("efficiency.png", dpi=200)

    print("Generated: accuracy.png, runtime.png, speedup.png, efficiency.png")
    print("Generated: results_with_metrics.csv")


if __name__ == "__main__":
    main()
