import torch
from torch import nn
import lightning as L
import torch.optim as optim
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryFBetaScore,
    F1Score,
    Precision,
    Recall,
    BinaryAveragePrecision,
)


class FraudNN(L.LightningModule):
    def __init__(
        self,
        input_features,
        hidden_units=128,
        num_layers=3,
        dropout_rate=0.3,
        learning_rate=0.001,
        output_features=1,
        threshold=0.5,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.threshold = threshold
        self.learning_rate = learning_rate

        # ✅ Add average="micro" for epoch-level aggregation (KEY FIX)
        self.train_metrics = MetricCollection(
            {
                "f2": BinaryFBetaScore(beta=2.0, zero_division=0),
                "recall": Recall(task="binary", average="micro", zero_division=0),
                "precision": Precision(task="binary", average="micro"),
                "f1": F1Score(task="binary", average="micro"),
                "ap": BinaryAveragePrecision(thresholds=None),
            },
            prefix="train_",
        )

        self.val_metrics = MetricCollection(
            {
                "f2": BinaryFBetaScore(beta=2.0, zero_division=0),
                "recall": Recall(task="binary", average="micro", zero_division=0),
                "precision": Precision(task="binary", average="micro"),
                "f1": F1Score(task="binary", average="micro"),
                "ap": BinaryAveragePrecision(thresholds=None),
            },
            prefix="val_",
        )

        self.test_metrics = MetricCollection(
            {
                "f2": BinaryFBetaScore(beta=2.0, zero_division=0),
                "recall": Recall(task="binary", average="micro", zero_division=0),
                "precision": Precision(task="binary", average="micro"),
                "f1": F1Score(task="binary", average="micro"),
                "ap": BinaryAveragePrecision(thresholds=None),
            },
            prefix="test_",
        )

        # Input Layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Hidden Layers
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_units, hidden_units),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(num_layers)
            ]
        )

        # Output Layer
        self.output_layer = nn.Linear(hidden_units, output_features)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()
        loss = self.criterion(logits, y.float())
        self.train_metrics.update(probs, y.int())

        # if batch_idx % 128 == 0:
        #     # Prediction Distribition
        #     unique_preds, counts = torch.unique(preds, return_counts=True)
        #     print(
        #         f"Training Batch {batch_idx}: Predictions distribution: {dict(zip(unique_preds.cpu().numpy(), counts.cpu().numpy()))}"
        #     )

        #     # Ground Truth Distribution
        #     # Ground truth distribution (ADD THIS)
        #     unique_y, counts_y = torch.unique(y, return_counts=True)
        #     print(
        #         f"Training Batch {batch_idx}: Ground truth distribution: {dict(zip(unique_y.cpu().numpy(), counts_y.cpu().numpy()))}\n"
        #     )

        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()
        loss = self.criterion(logits, y.float())
        self.val_metrics.update(probs, y.int())

        # if batch_idx % 128 == 0:
        #     # Predictions distribution
        #     unique_preds, counts = torch.unique(preds, return_counts=True)
        #     print(
        #         f"Validation Batch {batch_idx}: Predictions distribution: {dict(zip(unique_preds.cpu().numpy(), counts.cpu().numpy()))}"
        #     )

        #     # Ground truth distribution (ADD THIS)
        #     unique_y, counts_y = torch.unique(y, return_counts=True)
        #     print(
        #         f"Validation Batch {batch_idx}: Ground truth distribution: {dict(zip(unique_y.cpu().numpy(), counts_y.cpu().numpy()))}\n"
        #     )

        # batch_metrics = self.val_metrics.compute()
        # print(f"Validation Batch {batch_idx}: Batch metrics: {batch_metrics}")

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()
        loss = self.criterion(logits, y.float())
        self.test_metrics.update(probs, y.int())  # reuse test metrics for test

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
