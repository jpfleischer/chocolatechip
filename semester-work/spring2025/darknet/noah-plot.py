#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

def collect_records(base_dir: str) -> pd.DataFrame:
    records = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if not f.endswith(".csv"):
                continue
            if "benchmark__" not in f:
                continue
            csv_path = os.path.join(root, f)
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                # Some CSVs can have odd formatting; fallback parser
                df = pd.read_csv(csv_path, engine="python")
            # Expect single-row benchmark CSVs; but handle multi-row just in case
            for _, row in df.iterrows():
                # Prefer COCO AP50-95 (%), fallback to mAP50-95 (%)
                ap_col_candidates = [
                    "COCO AP50-95 (%)",
                    "mAP50-95 (%)",
                    "mAP50-95 (%) "  # sometimes trailing space happens
                ]
                ap_val = None
                for col in ap_col_candidates:
                    if col in df.columns:
                        ap_val = pd.to_numeric(row[col], errors="coerce")
                        break
                if ap_val is None or pd.isna(ap_val):
                    continue

                # YOLO version from the CSV ("YOLO Template"), fallback to directory name
                yolo = row["YOLO Template"] if "YOLO Template" in df.columns else os.path.basename(os.path.dirname(root))

                # Validation fraction from CSV, fallback to guessing from path
                val_frac = None
                if "Val Fraction" in df.columns:
                    val_frac = pd.to_numeric(row["Val Fraction"], errors="coerce")
                if val_frac is None or pd.isna(val_frac):
                    # Try to infer from directory segments like "val10", "val15", etc.
                    parts = root.split(os.sep)
                    val_tokens = [p for p in parts if p.startswith("val")]
                    if val_tokens:
                        try:
                            val_frac = float(val_tokens[0].replace("val", "")) / 100.0
                        except Exception:
                            val_frac = None

                if val_frac is None or pd.isna(val_frac):
                    # If still missing, skip this file
                    continue

                records.append({
                    "yolo": str(yolo),
                    "val_frac": float(val_frac),
                    "ap5095": float(ap_val),
                    "source_csv": csv_path
                })
    return pd.DataFrame.from_records(records)

def make_plot(df: pd.DataFrame, out_path: str):
    # Prepare readable validation ratio labels, e.g., "10%", "15%"
    df = df.copy()
    df["val_pct"] = (df["val_frac"] * 100).round().astype(int).astype(str) + "%"

    # Ordering of x-axis by numeric val fraction
    order = [f"{int(x)}%" for x in sorted(df["val_frac"].unique() * 100)]
    # Consistent hue order by YOLO template name
    hue_order = sorted(df["yolo"].unique())

    plt.figure(figsize=(10, 6))
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
        ax = sns.boxplot(data=df, x="val_pct", y="ap5095", hue="yolo", order=order, hue_order=hue_order)
        # Overlay means as diamonds
        means = df.groupby(["val_pct", "yolo"], as_index=False)["ap5095"].mean()
        sns.pointplot(
            data=means, x="val_pct", y="ap5095", hue="yolo",
            order=order, hue_order=hue_order, dodge=0.4, join=False,
            markers="D", scale=0.9, errwidth=0, linestyles="", palette="dark"
        )
        # Remove duplicated legend from the overlay
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(hue_order)], labels[:len(hue_order)], title="YOLO version")
    else:
        # Fallback without seaborn: simple grouped boxplots per (val_pct, yolo)
        # Build positions for grouped boxes
        groups = sorted([(vp, y) for vp in order for y in hue_order])
        data_by_group = {}
        for vp, y in groups:
            data_by_group[(vp, y)] = df[(df["val_pct"] == vp) & (df["yolo"] == y)]["ap5095"].tolist()

        # Compute positions
        x_positions = []
        plot_data = []
        tick_positions = []
        tick_labels = order
        group_width = 0.8
        dodge = group_width / (len(hue_order) + 1)
        base = 1
        for vp in order:
            tick_positions.append(base + group_width / 2)
            for i, y in enumerate(hue_order):
                x = base + i * dodge
                x_positions.append(x)
                plot_data.append(data_by_group[(vp, y)])
            base += 1

        bp = plt.boxplot(plot_data, positions=x_positions, widths=dodge * 0.9, patch_artist=True)
        # Color boxes by hue
        import itertools
        colors = plt.cm.tab10.colors
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(hue_order)])
        # Add mean markers
        means = df.groupby(["val_pct", "yolo"], as_index=False)["ap5095"].mean()
        base = 1
        idx = 0
        for vp in order:
            for i, y in enumerate(hue_order):
                m = means[(means["val_pct"] == vp) & (means["yolo"] == y)]["ap5095"]
                if len(m):
                    plt.scatter(base + i * dodge, float(m.iloc[0]), marker="D", color="black", s=40, zorder=3)
                idx += 1
            base += 1
        plt.xticks(tick_positions, tick_labels)
        # Simple legend
        for i, y in enumerate(hue_order):
            plt.scatter([], [], marker="s", color=colors[i % len(hue_order)], label=y)
        plt.legend(title="YOLO version")

    plt.title("COCO AP50-95 (%) by YOLO version and validation ratio")
    plt.xlabel("Validation ratio")
    plt.ylabel("COCO AP50-95 (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved box-and-whisker plot to: {out_path}")

def main():
    default_base = "/home/noah/chocolatechip/semester-work/spring2025/darknet/artifacts/outputs/LegoGearsDarknet"
    parser = argparse.ArgumentParser(description="Box-and-whisker plot of COCO AP50-95 grouped by YOLO version and validation ratio.")
    parser.add_argument("--base-dir", type=str, default=default_base, help="Base outputs directory to scan for benchmark CSVs.")
    parser.add_argument("--out", type=str, default="coco_ap5095_boxplot_by_yolo_val.png", help="Path to save the plot image.")
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        raise SystemExit(f"Base directory not found: {args.base_dir}")

    df = collect_records(args.base_dir)
    if df.empty:
        raise SystemExit("No benchmark CSVs found with the required columns.")

    # Basic sanity check printout
    summary = df.groupby(["yolo", "val_frac"]).agg(n=("ap5095", "size"), mean_ap=("ap5095", "mean")).reset_index()
    print("Found results:")
    for _, r in summary.iterrows():
        print(f"  YOLO={r['yolo']}, val={r['val_frac']:.2f}, n={int(r['n'])}, mean AP50-95={r['mean_ap']:.3f}")

    make_plot(df, args.out)

if __name__ == "__main__":
    main()