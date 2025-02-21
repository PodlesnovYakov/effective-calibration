from typing import Tuple, Dict, Any, Optional
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
    
    
class OUModel(pl.LightningModule):
    """PyTorch Lightning module for Ornstein-Uhlenbeck process parameter estimation.
    
       :param learning_rate: Initial learning rate for optimizer
       :param loss_fn: MSE loss function for parameter estimation
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3
    ):
        """Initialize the OU model with configurable parameter ranges.
           
           :param learning_rate: Initial learning rate for Adam optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.MSELoss()

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step with logging.
        
            :param batch: Tuple of (input sequences, target parameters)
            :param batch_idx: Batch index
            :return: Loss value
        """
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> None:
        """Validation step with epoch-level logging."""
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> None:
        """Test step with loss calculation."""
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("test_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
