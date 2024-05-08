import torch
from typing import *
from lightning import LightningModule


class AutoEncoder(torch.nn.Module):
    def __init__(self, str_len, hiddens: List[int], normalize: bool = False):
        super(AutoEncoder, self).__init__()

        # use a loop to create the encoder and decoder from hiddens
        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()

        if normalize:
            self.encoder.add_module("inp_norm", torch.nn.functional.sigmoid())

        self.encoder.add_module("input", torch.nn.Linear(str_len, hiddens[0]))

        # encoder
        for i in range(len(hiddens) - 1):
            self.encoder.add_module(
                f"enc_{i}", torch.nn.Linear(hiddens[i], hiddens[i + 1])
            )
            self.encoder.add_module(f"enc_{i}_relu", torch.nn.ReLU())

        # decoder
        for i in range(len(hiddens) - 1, 0, -1):
            self.decoder.add_module(
                f"dec_{i}", torch.nn.Linear(hiddens[i], hiddens[i - 1])
            )
            self.decoder.add_module(f"dec_{i}_relu", torch.nn.ReLU())

        self.decoder.add_module("output", torch.nn.Linear(hiddens[0], str_len))

        if normalize:
            self.decoder.add_module("out_norm", torch.nn.functional.inverse_sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoEncoderLightning(LightningModule):
    def __init__(self, str_len, hiddens: List[int]):
        super(AutoEncoderLightning, self).__init__()
        self.model = AutoEncoder(str_len, hiddens)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["string"]
        lang_labels = batch["language"]
        y_hat = self.model(x)
        y = x
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def _eval_step(self, batch, batch_idx, prefix):
        x = batch["string"]
        lang_labels = batch["language"]
        y_hat = self.model(x)
        y = x
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log(f"{prefix}_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
