import unittest
import torch
import torch.nn as nn

from src.models.factory import build_model


class TestModelFactory(unittest.TestCase):

    def test_build_model_binary(self):
        """Verify build_model instantiates EfficientNet-B0 with 2 output logits for binary stage."""
        model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
        self.assertIsInstance(model, nn.Module)

        dummy_input = torch.randn(2, 3, 224, 224)
        outputs = model(dummy_input)

        self.assertEqual(outputs.shape, (2, 2))

    def test_build_model_fine(self):
        """Verify build_model instantiates EfficientNet-B0 with 38 output logits for fine-grained stage."""
        model = build_model("efficientnet_b0", num_classes=38, pretrained=False)
        dummy_input = torch.randn(4, 3, 224, 224)
        outputs = model(dummy_input)

        self.assertEqual(outputs.shape, (4, 38))

    def test_model_backward_pass(self):
        """Verify that gradients propagate through the entire architecture."""
        model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_target = torch.tensor([0, 1], dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_target)
        loss.backward()

        has_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.norm(param.grad) > 0:
                has_grad = True
                break

        self.assertTrue(has_grad, "Model parameters should have non-zero gradients after backward pass")


if __name__ == "__main__":
    unittest.main()
