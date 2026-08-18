"""
Evaluate a saved checkpoint on the held-out TEST split -- never touched by
training or checkpoint selection, so this is the honest generalization
number (val macro_f1 during training is a slightly optimistic estimate,
since the best-epoch checkpoint was picked based on it).

Usage:
    python -m src.inference.evaluate_checkpoint --config configs/config.yaml --stage stage1
    python -m src.inference.evaluate_checkpoint --config configs/config.yaml --stage stage2
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data.augmentations import get_val_transforms
from src.data.dataset import LeafDataset
from src.models.factory import build_model
from src.train.engine import evaluate
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--stage", type=str, required=True, choices=["stage1", "stage2"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage_cfg = cfg[args.stage]
    label_col = stage_cfg["label_col"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(cfg["checkpoints_dir"]) / stage_cfg["checkpoint_name"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} -- train {args.stage} first.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_macro_f1={checkpoint['val_macro_f1']:.4f})")

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    splits_dir = Path(cfg["data"]["splits_dir"])
    test_ds = LeafDataset(splits_dir / "test.csv", label_col=label_col, transforms=get_val_transforms(cfg))
    # Force the test set to use the SAME class-index mapping the model was trained with,
    # not a mapping freshly derived from test.csv (which could differ in class order/coverage)
    test_ds.class_to_idx = class_to_idx

    model = build_model(checkpoint["backbone"], num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_cfg = cfg["train"]
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()  # unweighted for a clean, reportable test number
    use_amp = train_cfg["mixed_precision"] and device.type == "cuda"

    print(f"Evaluating on TEST set ({len(test_ds)} images, never seen during training)...")
    metrics = evaluate(model, test_loader, criterion, device, use_amp)

    print(f"\nTest loss:      {metrics['loss']:.4f}")
    print(f"Test accuracy:  {metrics['accuracy']:.4f}")
    print(f"Test macro-F1:  {metrics['macro_f1']:.4f}")

    if args.stage == "stage2":
        from src.train.report import generate_reports  # only needed here, keeps stage1 eval dependency-light

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        reports_dir = Path(cfg["reports_dir"])
        generate_reports(metrics["labels"], metrics["preds"], idx_to_class, reports_dir / "test")
        print(f"\nPer-class report + confusion matrix -> {reports_dir / 'test'}")


if __name__ == "__main__":
    main()
