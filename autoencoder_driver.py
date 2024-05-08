# %%
from scripts.model_def import TransformerPredictor
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# %load_ext autoreload
# %autoreload 2


# %%


def name_of_run(nheads, model_dim, num_classes, is_linear):
    return f"nheads={nheads}_model_dim={model_dim}_num_classes={num_classes}_is_linear={is_linear}"


# %%

from scripts.dataset_scripts.auto_encoder_data import create_dataset_splits
%ls

train, val, test = create_dataset_splits("data", max_len=20)

train[10]

# %%
from torch.utils.data import DataLoader
train_loader = DataLoader(train, batch_size=256, shuffle=True)
val_loader = DataLoader(val, batch_size=64)
test_loader = DataLoader(test, batch_size=64)

#%%

next(iter(train_loader))['language'].shape

#%%

config = {
    "experiment_type": "autoencoder",
    "str_len": 20,
    "hiddens": [16, 8, 16],
    "lr": 1e-3,
}


# wandb logger

run_name = "autoencoder_20_16_8_16_20"
logger = WandbLogger(project="MGM", name=run_name, config=config)

# early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)

# trainer
trainer = Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[early_stopping],
)

# model
from scripts.autoencoder import AutoEncoderLightning

model = AutoEncoderLightning(str_len=20, hiddens=[16, 8, 16])

# train
trainer.fit(model, train_loader, val_loader)

# test
trainer.test(model, test_loader)

# %%

# finish wandb experiment
logger.experiment.finish()
