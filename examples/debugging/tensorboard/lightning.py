import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
dataset = torch.utils.data.Subset(dataset, range(1000))
train, val = random_split(dataset, [800, 200])

autoencoder = LitAutoEncoder()
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    filepath="/state/checkpoints/sample-mnist-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
)
trainer = pl.Trainer(
    default_root_dir="/opt/ml/output/tensorboard",
    checkpoint_callback=checkpoint_callback,
)
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
