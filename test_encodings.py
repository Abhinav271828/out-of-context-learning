

# %%

# load lightning module
from lightning import LightningModule

%load_ext autoreload
%autoreload 2


# %%

# load checkpoint
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from scripts.autoencoder import AutoEncoderLightning
from scripts.dataset_scripts.auto_encoder_data import create_dataset_splits
from torch.utils.data import DataLoader

# load pretrained checkpoint

# model = AutoEncoderLightning.load_from_checkpoint("MGM/rh0h2m8h/checkpoints/epoch=17-step=313380.ckpt")
model = AutoEncoderLightning.load_from_checkpoint("MGM/5a3e1n10/checkpoints/epoch=18-step=330942.ckpt")
encoder = model.model.encoder


# %%

from sklearn.linear_model import LogisticRegression

# load data
train, val, test = create_dataset_splits("data", max_len=20)

# %%

# get data of a particular class from train
train_data = list(train)

# sample a subset of the data
train_data = train_data[:2000]

# %%

# make random strings of letters of length 20
import random
import string

def random_string(length, alphabet):
    length_ef = random.randint(1, length)
    print(alphabet)
    unpadded = [random.choice(alphabet[:-1]) for _ in range(length_ef)]
    padded = unpadded + [len(alphabet)-1] * (length - len(unpadded))
    print(padded)
    return torch.tensor(padded, dtype=torch.float32)

encoder.to("cpu")

# %%

# make 1000 random strings
strings = [random_string(20, [0, 1, 2, 3]) for _ in range(2000)]


Y = [1]*2000 + [0] * 2000
X = [encoder(x["string"]).detach().cpu().numpy() for x in train_data] + [encoder(x).detach().cpu().numpy() for x in strings]


# %%

# make train test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# %%

# train a logistic regression model
clf = LogisticRegression(random_state=0, max_iter=2000, fit_intercept=False).fit(X, Y)

# %%

# evaluate
clf.score(X_test, Y_test)

# get classification report
from sklearn.metrics import classification_report

print(classification_report(Y_test, clf.predict(X_test)))


# %%

# get the coef array 
import numpy as np
coef = clf.coef_.reshape(-1)

# turn into a pytorch linear layer
from torch.nn import Linear
linear = Linear(16, 1, bias=False)

# set the weights
linear.weight.data = torch.tensor(coef.reshape(1, -1), dtype=torch.float32)

print(coef.shape)
torch.save(linear, 'classify_between_trash_and_regula.pt')