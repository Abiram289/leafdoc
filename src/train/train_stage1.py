"""
Stage 1: binary Healthy vs Diseased classifier.

Fast to train (2 classes, strong signal) -- this is meant to get a working
end-to-end loop (data -> model -> checkpoint -> metrics) proven out before
Stage 2's harder 38-class problem.

Usage:
    python -m src.train.train_stage1 --config configs/config.yaml
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import LeafDataset
from src.models.factory import build_model
from src.train.engine import evaluate, train_one_epoch
from src.utils.config import load_config
from src.utils.seed import set_seed


def compute_class_weights(dataset: LeafDataset) -> torch.Tensor:
    """Inverse-frequency weights so the majority class (diseased, ~72%) doesn't
    dominate the loss -- PlantVillage's binary split is imbalanced."""
    counts = dataset.df[dataset.label_col].value_counts()
    counts = counts.reindex(sorted(dataset.class_to_idx, key=dataset.class_to_idx.get))
    weights = 1.0 / counts.values
    weights = weights / weights.sum() * len(weights)  # normalize to mean 1
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits_dir = Path(cfg["data"]["splits_dir"])
    stage_cfg = cfg["stage1"]
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
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images, classes: {train_ds.class_to_idx}")

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
    print(f"Class weights ({list(train_ds.class_to_idx.keys())}): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    epochs = train_cfg["epochs_stage1"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = train_cfg["mixed_precision"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    checkpoints_dir = Path(cfg["checkpoints_dir"])
    logs_dir = Path(cfg["logs_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "stage1_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "lr"])

    best_f1 = -1.0
    ckpt_path = checkpoints_dir / stage_cfg["checkpoint_name"]

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


if __name__ == "__main__":
    main()
