#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from plot_common import get_ordered_yolos, normalize_dataset_name

DEFAULT_BASES = [
    Path("/home/artur/chocolatechip/semester-work/spring2025/darknet"),
    Path("/home/artur/chocolatechip/semester-work/fall2025/ultralytics"),
]

def find_cm_files(root: Path):
    return list(root.rglob("confusion_matrix.json"))

def load_json(p):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def avg_f1(values):
    return sum(values) / len(values) if values else float("nan")

def clean_yolo_name(yolo_template):
    # Remove "yolo" prefix
    cleaned = yolo_template.lower().replace("yolo", "")
    # Remove anything after the dot
    if "." in cleaned:
        cleaned = cleaned.split(".")[0]
    return cleaned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="", help="optional root directory")
    args = parser.parse_args()
    
    roots = [Path(args.root)] if args.root else DEFAULT_BASES
    scores = defaultdict(list)
    
    for root in roots:
        for cm_path in find_cm_files(root):
            dir_path = cm_path.parent
            csv_files = list(dir_path.glob("*.csv"))
            if not csv_files:
                continue
            
            csv_path = csv_files[0]
            df_csv = pd.read_csv(csv_path, dtype=str)
            
            if "YOLO Template" not in df_csv.columns or "Backend" not in df_csv.columns:
                continue
            
            yolo_template = df_csv["YOLO Template"].dropna().iloc[0]
            val_fraction = df_csv["Val Fraction"].dropna().iloc[0] if "Val Fraction" in df_csv.columns else "unknown"
            
            cleaned_yolo = clean_yolo_name(yolo_template)
            combined_key = f"{cleaned_yolo} / {val_fraction}"
            
            data = load_json(cm_path)
            for rec in data.get("per_class", []):
                key = (rec["class"], combined_key)
                scores[key].append(rec["f1"])
    
    rows = [
        {"class": k[0], "YOLO_Val": k[1], "avg_f1": avg_f1(v), "count": len(v)}
        for k, v in scores.items()
    ]
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("per_class_yolo_f1.csv", index=False)
    
    cars_classes = ["motorbike", "car", "truck", "bus", "pedestrian"]
    leather_classes = ["color", "cut", "fold", "glue", "poke"]
    lego_classes = ["red light", "pin", "center", "small gear", "medium gear"]
    
    groups = {
        "cars": cars_classes,
        "leather": leather_classes,
        "lego": lego_classes,
    }
    
    output_dir = Path("heatmaps")
    output_dir.mkdir(exist_ok=True)
    
    sns.set(style="whitegrid", font_scale=1.15)
    
    # Define YOLO version order
    yolo_order = ["v4-tiny", "v7-tiny", "11n", "11s"]
    val_order = ["0.1", "0.15", "0.2", "0.8"]
    
    def sort_key(col):
        parts = col.split(" / ")
        if len(parts) != 2:
            return (999, 999)
        yolo_ver = parts[0]
        val_frac = parts[1]
        yolo_idx = yolo_order.index(yolo_ver) if yolo_ver in yolo_order else 999
        val_idx = val_order.index(val_frac) if val_frac in val_order else 999
        return (yolo_idx, val_idx)
    
    for name, classes in groups.items():
        sub = df[df["class"].isin(classes)]
        if sub.empty:
            continue
        
        heatmap_data = sub.pivot(index="class", columns="YOLO_Val", values="avg_f1")
        
        present_cols = heatmap_data.columns.tolist()
        ordered_cols = sorted(present_cols, key=sort_key)
        heatmap_data = heatmap_data[ordered_cols]
        
        class_means = heatmap_data.mean(axis=1).sort_values(ascending=False)
        heatmap_data = heatmap_data.loc[class_means.index]
        
        plt.figure(figsize=(max(12, len(ordered_cols)*0.8), 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn",
                    cbar_kws={"label": "Avg F1", "shrink": 0.7}, vmin=0, vmax=1, square=True)
        plt.yticks(rotation=0)
        plt.xlabel("")
        plt.ylabel("Class")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        out_path = output_dir / f"{name}_heatmap.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.show()

if __name__ == "__main__":
    main()