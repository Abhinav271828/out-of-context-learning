# %%
from scripts.model_def import MembershipLightning
from dataset import MembershipFewShot
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

%load_ext autoreload
%autoreload 2


# %%


def name_of_run(nheads, model_dim, num_classes, is_linear):
    return f"membership-autoencoder_no_clf-nheads={nheads}_model_dim={model_dim}_num_classes={num_classes}_is_linear={is_linear}"


# def main():

# datasets
# %%
train_dataset = MembershipFewShot(10, 80000)
val_dataset = MembershipFewShot(10, 10000)
test_dataset = MembershipFewShot(10, 10000)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# model
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
clf = torch.load("classify_between_trash_and_regula.pt")

model = MembershipModelPlusAutoEncoderLightning(
    string_length=20,
    model_dim=17,
    num_classes=2,
    num_heads=1,
    lr=1e-3,
    autoencoder=encoder,
    # regression_func=clf,
    linear_attention=True,
)

# %%
# logger
run_name = name_of_run(1, 17, 1, True)
config = {
    "experiment_type": "membership_encoder_no_clf",
    "str_len": 20,
}
logger = WandbLogger(project="MGM", name=run_name, config=config)

# early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)

# trainer
trainer = Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[early_stopping],
)

# train
trainer.fit(model, train_loader, val_loader)

# test
trainer.test(model, test_loader)

# %%

logger.experiment.finish()


# if __name__ == "__main__":
#     main()

# %%
