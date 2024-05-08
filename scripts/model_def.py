from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from typing import *
import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall
from lightning import LightningModule


def scaled_dot_product(q, k, v, mask=None, linear: bool = True):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    if linear:
        attention = attn_logits
    else:
        attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, is_linear: bool = True):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.is_linear = is_linear

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(
            q, k, v, mask=mask, linear=self.is_linear
        )
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


def construct_MLP(hidden_dims: List[int], activation: nn.Module = nn.ReLU):
    layers = []
    for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation())
    return nn.Sequential(*layers)


class PositionalEncoding(nn.Module):
    def __init__(self, dmodel, max_len=5000):
        """Positional Encoding.

        Args:
            dmodel: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(
            (
                int(max_len),
                dmodel,
            )
        )
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # repeat pe on the 0 dim
        pe = self.pe.repeat(x.size(0), 1, 1)
        x = x + pe[:, : x.size(1), :]
        return x


class NTPModel(torch.nn.Module):

    def __init__(
        self,
        ntoken: int,
        dmodel: int,
        nhead: int,
        nlayers: int,
        hiddens: List[int],
        linear_attention: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.dmodel = dmodel
        self.is_linear = linear_attention

        self.pos_encoder = PositionalEncoding(dmodel)

        self.multihead_attn = MultiheadAttention(
            input_dim=dmodel,
            embed_dim=dmodel,
            num_heads=nhead,
            is_linear=linear_attention,
        )

        self.embedding = nn.Embedding(ntoken, dmodel)

        self.mlp = construct_MLP([dmodel] + hiddens + [dmodel])

        self.linear = nn.Linear(dmodel, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.dmodel)
        src = self.pos_encoder(src)
        src = self.mlp(src)
        if mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            mask = nn.Transformer.generate_square_subsequent_mask(src.shape[-2]).to(
                src.device
            )
        output = self.multihead_attn(src, mask=mask)
        output = self.linear(output)
        return output


import torch.optim as optim
import numpy as np


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class NTPLightning(LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        lr,
        hiddens=[],
        linear_attention=True,
        dropout=0.0,
        input_dropout=0.0,
        warmup=100,
        max_iters=1000,
    ):
        """NTPLightning.

        Args:
            input_dim: Hidden dimensionality of the input
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
        self.hparams.num_layers = num_layers
        self.hparams.lr = lr
        self.hparams.linear_attention = linear_attention
        self.hparams.dropout = dropout
        self.hparams.input_dropout = input_dropout
        self.hparams.input_dim = input_dim
        self.hparams.num_classes = num_classes
        self.hparams.hiddens = hiddens
        self.hparams.warmup = warmup
        self.hparams.max_iters = max_iters
        self._create_model()

    def _create_model(self):
        self.transformer = NTPModel(
            dmodel=self.hparams.model_dim,
            nhead=self.hparams.num_heads,
            nlayers=self.hparams.num_layers,
            hiddens=self.hparams.hiddens,
            dropout=self.hparams.dropout,
            ntoken=self.hparams.num_classes,
            linear_attention=self.hparams.linear_attention,
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.transformer(x, mask=mask)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

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

    def training_step(self, batch, batch_idx):
        # next token prediction
        # batch is (b, s)
        x = batch[:, :-1]
        y = batch[:, 1:]

        # forward pass
        y_hat = self(x)
        b, s, d = y_hat.size()
        loss = F.cross_entropy(y_hat.reshape(-1, d), y.reshape(-1))

        self.log("train_loss", loss)
        return loss

    def eval_steps(self, batch, batch_idx):
        x = batch[:, :-1]
        y = batch[:, 1:]

        y_hat = self(x)
        b, s, d = y_hat.size()
        loss = F.cross_entropy(y_hat.reshape(-1, d), y.reshape(-1))

        accuracy = Accuracy(task="multiclass", num_classes=4).to(self.device)
        precision = Precision(task="multiclass", num_classes=4).to(self.device)
        recall = Recall(task="multiclass", num_classes=4).to(self.device)

        acc = accuracy(y_hat.reshape(-1, d).argmax(dim=-1), y.reshape(-1))
        prec = precision(y_hat.reshape(-1, d).argmax(dim=-1), y.reshape(-1))
        rec = recall(y_hat.reshape(-1, d).argmax(dim=-1), y.reshape(-1))

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

class MembershipModel(torch.nn.Module):
    def __init__(
        self,
        str_len: int,
        dmodel: int,
        nhead: int,
        hiddens: List[int],
        linear_attention: bool = True,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.dmodel = dmodel
        self.is_linear = linear_attention

        self.multihead_attn = MultiheadAttention(
            input_dim=dmodel,
            embed_dim=dmodel,
            num_heads=nhead,
            is_linear=linear_attention,
        )

        self.mlp = construct_MLP([str_len+1] + hiddens + [dmodel])

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
        src = self.mlp(src)
        # src : [b, n_e+1, d_model]
        output = self.multihead_attn(src)
        # output : [b, n_e+1, d_model]
        # We interpret the last element of the vectors to be the label
        # and return the label of the test token.
        return output[:, -1, -1]

class MembershipLightning(LightningModule):
    def __init__(
        self,
        string_length,
        model_dim,
        num_classes,
        num_heads,
        lr,
        hiddens=[],
        linear_attention=True,
        input_dropout=0.0,
        warmup=100,
        max_iters=1000,
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
        self.hparams.hiddens = hiddens
        self.hparams.warmup = warmup
        self.hparams.max_iters = max_iters
        self._create_model()

    def _create_model(self):
        self.transformer = MembershipModel(
            str_len=self.hparams.string_length,
            dmodel=self.hparams.model_dim,
            nhead=self.hparams.num_heads,
            hiddens=self.hparams.hiddens,
            linear_attention=self.hparams.linear_attention,
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

    def training_step(self, batch, batch_idx):
        # next token prediction
        # batch is (b, s)
        x = batch[0]
        y = batch[1]

        # forward pass
        y_hat = self(x) # y_hat : [bz]
        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def eval_steps(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]

        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        accuracy = Accuracy(task="binary").to(self.device)
        precision = Precision(task="binary").to(self.device)
        recall = Recall(task="binary").to(self.device)

        acc = accuracy(y_hat, y)
        prec = precision(y_hat, y)
        rec = recall(y_hat, y)

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
