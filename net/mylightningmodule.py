import pytorch_lightning as pl
import torch
from .network import Network
from parameters import *


class MyLightningModule(pl.LightningModule):

    def __init__(self, model: Network, lr):
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=LEARNING_RATE_DECAY, gamma=GAMMA)
        self.loss_func = torch.nn.functional.cross_entropy

    """
    used for inference only
    """
    def forward(self, x):
        return self.model.forward(x)  # regular forward of the network

    """
    mandatory function as part of lightning module
    """
    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def one_step(self, batch,):
        x, y = batch
        y_flattened = y.flatten()
        outs = self.forward(x)
        loss = self.loss_func(outs, y_flattened)  # cross-entropy loss
        predictions = outs.argmax(1)
        amount_of_right_predictions = float(predictions.eq(y_flattened).sum())
        accuracy = amount_of_right_predictions / len(outs)
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, acc = self.one_step(batch)
        return {'loss': loss, 'progress_bar': {'acc': acc}}

    def validation_step(self, batch, batch_nb):
        loss, acc = self.one_step(batch)
        return {'val_loss': loss, "val_acc": acc}

    def test_step(self, batch, batch_nb):
        loss, acc = self.one_step(batch)
        return {'test_loss': loss, "test_acc": acc}

