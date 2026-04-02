import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import R2Score, MeanSquaredError

class BiomassRegressor(pl.LightningModule):
    def __init__(self, input_dim:int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # MLP Architecture for Regression
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Metrik (R-Squared and RMSE)
        self.r2_metric = R2Score()
        self.mse_metric = MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        
        # Make sure y has shape (batch_size, 1) to match the model output.
        y = y.view(-1, 1).float() 
        
        # Forward pass
        preds = self(x)
        
        # Culculate Loss (MSE)
        loss = self.loss_fn(preds, y)
        
        # Calculate the R2 Metric
        r2 = self.r2_metric(preds, y)
        
        return loss, r2

    def training_step(self, batch, batch_idx):
        loss, r2 = self._common_step(batch, batch_idx)
        # Log loss and r2
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, r2 = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, r2 = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_r2', r2, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)