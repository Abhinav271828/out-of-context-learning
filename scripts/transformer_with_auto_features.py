from .model_def import (
    MultiheadAttention,
    CosineWarmupScheduler,
)
import torch
from torch import Tensor
from typing import *
from lightning import LightningModule
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchmetrics import Accuracy, Precision, Recall
from scripts.dataset_scripts.auto_encoder_data import create_dataset_splits
from scripts.autoencoder import AutoEncoderLightning
from scripts.dataset_scripts.auto_encoder_data import create_dataset_splits
from torch.utils.data import DataLoader
import random


class MembershipModelPlusAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        str_len: int,
        dmodel: int,
        nhead: int,
        autoencoder: torch.nn.Module,
        regression_func: Optional[torch.nn.Linear] = None,
        linear_attention: bool = True,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.dmodel = dmodel
        self.is_linear = linear_attention  # d_model-1

        self.encoder = autoencoder
        self.regression_func = regression_func

        self.multihead_attn = MultiheadAttention(
            input_dim=dmodel,
            embed_dim=dmodel,
            num_heads=nhead,
            is_linear=linear_attention,
        )

        self.strlen = str_len

    def make_features(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, num_examples+1, str_len+1]``
                num_examples is the number of examples per task
                    the last example is the test sample
                str_len is the maximum length of an input string
                    the last position is the label
        """
        with torch.no_grad():
            src_in = src[:, :, :-1]  # [b, n_e+1, s_l]
            src_in = self.encoder(src_in)  # [b, n_e+1, d_model-1]
            if self.regression_func is not None:
                outs = self.regression_func(src_in)  # [b, n_e+1, 1]
            else:
                outs = torch.zeros(src.size(0), src.size(1), 1, device=src.device)

            return torch.cat([src_in, outs], dim=-1)  # [b, n_e+1, d_model]

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, num_examples+1, str_len+1]``
                num_examples is the number of examples per task
                    the last example is the test sample
                str_len is the maximum length of an input string
                    the last position is the label
        Returns:
            output Tensor of shape ``[batch_size]``
        """
        # src : [b, n_e+1, s_l+1]
        src = self.make_features(src)
        # src : [b, n_e+1, d_model]
        output = self.multihead_attn(src)
        # output : [b, n_e+1, d_model]
        # We interpret the last element of the vectors to be the label
        # and return the label of the test token.
        return output[:, -1, -1]


class MembershipModelPlusAutoEncoderLightning(LightningModule):
    def __init__(
        self,
        string_length,
        model_dim,
        num_classes,
        num_heads,
        lr,
        autoencoder,
        linear_attention=True,
        input_dropout=0.0,
        warmup=100,
        max_iters=1000,
        regression_func=None,
        att_object=None,
    ):
        """NTPLightning.

        Args:
            string_length: Length of strings forming samples in tasks
            model_dim: Hidden dimensionality to use inside the Transformer
            num_classes: Number of classes to predict per sequence element
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            num_layers: Number of encoder blocks to use.
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps. Usually between 50 and 500
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout: Dropout to apply inside the model
            input_dropout: Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self.hparams.model_dim = model_dim
        self.hparams.num_heads = num_heads
        self.hparams.lr = lr
        self.hparams.linear_attention = linear_attention
        self.hparams.input_dropout = input_dropout
        self.hparams.string_length = string_length
        self.hparams.num_classes = num_classes
        self.hparams.warmup = warmup
        self.hparams.max_iters = max_iters
        self.hparams.regression_func = regression_func
        self.autoencoder = autoencoder
        self.att_object = att_object
        self._create_model()

    def _create_model(self):
        self.transformer = MembershipModelPlusAutoEncoder(
            str_len=self.hparams.string_length,
            dmodel=self.hparams.model_dim,
            nhead=self.hparams.num_heads,
            linear_attention=self.hparams.linear_attention,
            autoencoder=self.autoencoder,
            regression_func=self.hparams.regression_func,
        )

    def forward(self, x):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.transformer(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def _compare_qkv(self):
        self.log("WHATEVER YOU WANT BANI")
        # TODO: Bani fill this up and log the error thing

    def training_step(self, batch, batch_idx):
        # next token prediction
        # batch is (b, s)
        x = batch[0]
        y = batch[1]

        # forward pass
        y_hat = self(x)  # y_hat : [bz]
        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss)
        self._compare_qkv()
        return loss

    def eval_steps(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]

        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        y_hat_preds = torch.where(y_hat > 0, 1.0, 0.0)

        accuracy = Accuracy(task="binary").to(self.device)
        precision = Precision(task="binary").to(self.device)
        recall = Recall(task="binary").to(self.device)

        acc = accuracy(y_hat_preds, y)
        prec = precision(y_hat_preds, y)
        rec = recall(y_hat_preds, y)

        return {
            "loss": loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
        }

    def validation_step(self, batch, batch_idx):
        metric_dict = self.eval_steps(batch, batch_idx)
        val_acc = metric_dict["accuracy"]
        val_prec = metric_dict["precision"]
        val_rec = metric_dict["recall"]
        val_loss = metric_dict["loss"]

        self.log("val_loss", val_loss)
        self.log("val_accuracy", val_acc)
        self.log("val_precision", val_prec)
        self.log("val_recall", val_rec)
        return val_loss

    def test_step(self, batch, batch_idx):
        metric_dict = self.eval_steps(batch, batch_idx)
        test_acc = metric_dict["accuracy"]
        test_prec = metric_dict["precision"]
        test_rec = metric_dict["recall"]
        test_loss = metric_dict["loss"]

        self.log("val_loss", test_loss)
        self.log("val_accuracy", test_acc)
        self.log("val_precision", test_prec)
        self.log("val_recall", test_rec)
        return test_loss
