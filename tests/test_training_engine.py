import unittest
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from src.train.engine import evaluate, train_one_epoch


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class TestTrainingEngine(unittest.TestCase):

    def test_engine_train_and_eval(self):
        """Verify train_one_epoch and evaluate execution loops on synthetic batches."""
        device = torch.device("cpu")
        model = ToyClassifier().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=False)

        # 8 dummy samples: (8, 3, 32, 32)
        x = torch.randn(8, 3, 32, 32)
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Test train_one_epoch
        initial_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=False,
        )
        self.assertIsInstance(initial_loss, float)
        self.assertGreater(initial_loss, 0)

        # Test evaluate
        metrics = evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            use_amp=False,
        )
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["macro_f1"] <= 1.0)
        self.assertEqual(len(metrics["preds"]), 8)
        self.assertEqual(len(metrics["labels"]), 8)


if __name__ == "__main__":
    unittest.main()
