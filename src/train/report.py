"""
Post-training diagnostics for Stage 2: a per-class F1 report and a
confusion matrix. Run once against the best checkpoint, not every epoch --
a 38x38 heatmap and full classification_report are too expensive to
regenerate every epoch and the per-epoch macro_f1 in the training log is
enough to track progress during training itself.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def generate_reports(labels: list[int], preds: list[int], idx_to_class: dict[int, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Per-class precision/recall/F1
    report_str = classification_report(labels, preds, target_names=class_names, digits=3, zero_division=0)
    report_path = out_dir / "stage2_classification_report.txt"
    report_path.write_text(report_str)
    print(f"Per-class report -> {report_path}")

    # Confusion matrix, normalized by true-class row so it reads as
    # "of images actually in this class, what fraction went where"
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stage 2 confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    cm_path = out_dir / "stage2_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix -> {cm_path}")

    # Also dump raw counts as CSV for anyone who wants to look up exact numbers
    csv_path = out_dir / "stage2_confusion_matrix.csv"
    header = "," + ",".join(class_names)
    lines = [header]
    for name, row in zip(class_names, cm):
        lines.append(f"{name}," + ",".join(str(v) for v in row))
    csv_path.write_text("\n".join(lines))
    print(f"Confusion matrix (raw counts) -> {csv_path}")
