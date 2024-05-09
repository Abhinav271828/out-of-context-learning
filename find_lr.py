from scripts.transformer_with_auto_features import MembershipModelPlusAutoEncoderLightning
from dataset import MembershipFewShot
import torch
from torch.utils.data import DataLoader

def find_lr(model_path, val_dataloader):
    model = MembershipModelPlusAutoEncoderLightning.load_from_checkpoint(model_path)

    y_GDs = []
    y_hats = []
    for batch in val_dataloader:
        x, _ = batch
        bz, num_samples, str_len = x[:, :-1, :-1].shape

        y_hat = model(x)
        y_hats.append(y_hat)

        input_samples = x[:, :-1, :-1].flatten(0, 1).unsqueeze(-2)
        input_labels = x[:, :-1, -1:].flatten(0, 1).unsqueeze(-1)

        Δw = torch.bmm(- input_labels, input_samples).reshape(bz, num_samples, str_len).mean(dim=1)

        y_GD = torch.bmm(x[:, -1:, :-1], Δw.view(bz, 20, 1)).squeeze()
        y_GDs.append(y_GD)
    
    y_GDs = torch.cat(y_GDs)
    y_hats = torch.cat(y_hats)

    lr = torch.dot(y_GDs, y_hats) / torch.dot(y_hats, y_hats)
    return lr

if __name__ == '__main__':
    ds = MembershipFewShot(10, 10000)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    print(find_lr('epoch=32-step=41250.ckpt', dl))