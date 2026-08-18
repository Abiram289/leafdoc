"""
Stage 2: fine-grained 38-class species+disease classifier.

Harder than Stage 1 -- more classes, more epochs, and per-class performance
matters (a model that's 99% overall but blind to one rare disease is a
real failure mode a single accuracy number hides). After training, the
best checkpoint is re-evaluated once to produce a per-class F1 report and
confusion matrix -- see src/train/report.py.

Usage:
    python -m src.train.train_stage2 --config configs/config.yaml

    # Quick smoke test (couple minutes, not a real training run) before
    # committing to the full ~2-4hr job -- confirms the loop, checkpointing,
    # and report generation all work end-to-end on a tiny slice of data:
    python -m src.train.train_stage2 --config configs/config.yaml --epochs 2 --max-samples 500
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import LeafDataset
from src.models.factory import build_model
from src.train.engine import evaluate, train_one_epoch
from src.train.report import generate_reports
from src.utils.config import load_config
from src.utils.seed import set_seed


def compute_class_weights(dataset: LeafDataset) -> torch.Tensor:
    """Inverse-frequency weights -- class sizes range from ~220 to ~1800+
    images in this split, and Stage 2's whole point is per-class recall,
    not overall accuracy, so the rare classes need the loss boost."""
    counts = dataset.df[dataset.label_col].value_counts()
    counts = counts.reindex(sorted(dataset.class_to_idx, key=dataset.class_to_idx.get))
    weights = 1.0 / counts.values
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs_stage2, e.g. for a smoke test")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="subsample train/val to at most this many images (stratified), for a fast smoke test"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits_dir = Path(cfg["data"]["splits_dir"])
    stage_cfg = cfg["stage2"]
    label_col = stage_cfg["label_col"]
    class_map_path = stage_cfg["class_map"]

    train_ds = LeafDataset(
        splits_dir / "train.csv", label_col=label_col,
        transforms=get_train_transforms(cfg), class_map_path=class_map_path,
    )
    val_ds = LeafDataset(
        splits_dir / "val.csv", label_col=label_col,
        transforms=get_val_transforms(cfg), class_map_path=class_map_path,
    )

    if args.max_samples:
        # Stratified subsample so all 38 classes still appear even in a tiny smoke test.
        # Uses explicit iteration + pd.concat rather than groupby().apply() -- pandas 3.0
        # changed apply() to no longer pass the grouping column into the function, which
        # would silently drop label_col from the result. Direct iteration always keeps
        # every original column regardless of pandas version.
        import pandas as pd

        def stratified_subsample(df, col, total_n, seed):
            per_class = max(2, total_n // df[col].nunique())
            parts = [g.sample(min(len(g), per_class), random_state=seed) for _, g in df.groupby(col)]
            return pd.concat(parts).reset_index(drop=True)

        train_ds.df = stratified_subsample(train_ds.df, label_col, args.max_samples, cfg["seed"])
        val_frac = len(val_ds.df) / (len(val_ds.df) + len(train_ds.df))
        val_max = max(int(args.max_samples * val_frac), train_ds.num_classes * 2)
        val_ds.df = stratified_subsample(val_ds.df, label_col, val_max, cfg["seed"])
        print(f"[smoke test] Subsampled to {len(train_ds)} train / {len(val_ds)} val images")
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images, num_classes: {train_ds.num_classes}")

    train_cfg = cfg["train"]
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True,
    )

    model = build_model(train_cfg["backbone"], num_classes=train_ds.num_classes).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs_stage2"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = train_cfg["mixed_precision"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    checkpoints_dir = Path(cfg["checkpoints_dir"])
    logs_dir = Path(cfg["logs_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "stage2_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "lr"])

    best_f1 = -1.0
    ckpt_path = checkpoints_dir / stage_cfg["checkpoint_name"]
    best_val_metrics = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | val_acc {val_metrics['accuracy']:.4f} | "
            f"val_macro_f1 {val_metrics['macro_f1']:.4f} | lr {current_lr:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, val_metrics["loss"], val_metrics["accuracy"], val_metrics["macro_f1"], current_lr]
            )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "backbone": train_cfg["backbone"],
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                },
                ckpt_path,
            )
            print(f"  -> new best (val_macro_f1={best_f1:.4f}), saved to {ckpt_path}")

    print(f"\nDone. Best val_macro_f1: {best_f1:.4f}. Checkpoint: {ckpt_path}")
    print(f"Per-epoch log: {log_path}")

    # Per-class F1 + confusion matrix from the best epoch's predictions
    # (already computed during that epoch's validation pass -- no need to re-run inference)
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    reports_dir = Path(cfg["reports_dir"])
    generate_reports(best_val_metrics["labels"], best_val_metrics["preds"], idx_to_class, reports_dir)


if __name__ == "__main__":
    main()
