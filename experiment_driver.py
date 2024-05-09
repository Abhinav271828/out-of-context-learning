# %%
from scripts.model_def import MembershipLightning
from dataset import MembershipFewShot
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

#%load_ext autoreload
#%autoreload 2


# %%


def name_of_run(nheads, model_dim, num_classes, is_linear):
    return f"membership-autoencoder_no_clf-nheads={nheads}_model_dim={model_dim}_num_classes={num_classes}_is_linear={is_linear}"


# First, we define datasets and dataloaders that sample from all the languages (including negative samples).
# %%
train_dataset = MembershipFewShot(10, 80000)
val_dataset = MembershipFewShot(10, 10000)
test_dataset = MembershipFewShot(10, 10000)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Then, we define the model that will train on these datasets.
# Note that the autoencoder is pretrained.
# %%
from scripts.transformer_with_auto_features import (
    MembershipModelPlusAutoEncoderLightning,
)
from scripts.autoencoder import AutoEncoderLightning
import torch

model = AutoEncoderLightning.load_from_checkpoint(
    "MGM/5a3e1n10/checkpoints/epoch=18-step=330942.ckpt"
)
encoder = model.model.encoder

model = MembershipModelPlusAutoEncoderLightning(
    string_length=20,
    model_dim=17,
    num_classes=2,
    num_heads=1,
    lr=1e-3,
    autoencoder=encoder,
    linear_attention=True,
)

# We define the logger and the trainer objects, and train and test the model.
# %%
run_name = name_of_run(1, 17, 1, True)
config = {
    "experiment_type": "membership_encoder_no_clf",
    "str_len": 20,
}
logger = WandbLogger(project="MGM", name=run_name, config=config)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)

trainer = Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[early_stopping],
)

trainer.fit(model, train_loader, val_loader)

trainer.test(model, test_loader)

# %%

logger.experiment.finish()
