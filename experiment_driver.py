# %%
from scripts.model_def import TransformerPredictor
from dataset import ShuffleDyck2
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# %load_ext autoreload
# %autoreload 2


# %%


def name_of_run(nheads, model_dim, num_classes, is_linear):
    return f"nheads={nheads}_model_dim={model_dim}_num_classes={num_classes}_is_linear={is_linear}"


# def main():

# datasets
# %%
train_dataset = ShuffleDyck2("../train_dyck_2.txt", percent=0.06)
val_dataset = ShuffleDyck2("../val_dyck_2.txt")
test_dataset = ShuffleDyck2("../test_dyck_2.txt")

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# model
# %%
model = TransformerPredictor(
    input_dim=4,
    model_dim=16,
    num_classes=4,
    num_heads=8,
    num_layers=1,
    hiddens=[8, 8],
    lr=1e-3,
    linear_attention=True,
)

# %%
# logger
run_name = name_of_run(8, 16, 4, True)
logger = WandbLogger(project="MGM", name=run_name)

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


# if __name__ == "__main__":
#     main()

# %%
