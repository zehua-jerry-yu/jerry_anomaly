from utils import *
from model import *
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os

from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    AsDiscrete,  
    RandFlipd, 
    RandRotate90d, 
    NormalizeIntensityd,
    RandCropByLabelClassesd,
    Resized,
)


def get_pcb_images():
    from PIL import Image
    PATH_DATA = "/root/autodl-tmp/kaggle/anomaly/pcb2/mixed"
    filenames = os.listdir(PATH_DATA)
    filenames = [f for f in filenames if f.endswith(".JPG")]
    filenames = [f for f in filenames if len(f) == 8]  # only normal samples
    all_images = []
    for filename in filenames:
        image = Image.open(os.path.join(PATH_DATA, filename))
        image = np.array(image)  # (H,W,C)
        image = np.moveaxis(image, 2, 0)  # (C,H,W)
        all_images.append(image)
    return all_images
    

class GetMask(object):
    # transform. given image, generate random mask.
    def __init__(self, k, nfolds):
        self.k = k
        self.nfolds = nfolds

    def __call__(self, sample):
        sample = sample['image']
        c, h, w = sample.shape
        assert h % self.k == 0
        assert w % self.k == 0
        hk = h // self.k
        wk = w // self.k
        mask = np.full((hk * wk,), False, dtype=bool)
        mask[:int(hk * wk / self.nfolds)] = True
        mask = np.random.permutation(mask)
        mask = mask.reshape(hk, wk)
        mask = np.repeat(mask, self.k, axis=0)
        mask = np.repeat(mask, self.k, axis=1)
        return {'image': sample, 'mask': mask}

    
def get_pcb_loaders(train_images, valid_images):
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 64

    transforms = Compose([
        NormalizeIntensityd(keys="image"),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        Resized(spatial_size=(1000, 1312), size_mode='all', keys="image"),
        GetMask(k=4, nfolds=5)
    ])
    train_ds = Dataset(data=train_images, transform=transforms)
    # TODO: validation should be non random. how??
    valid_ds = Dataset(data=valid_images, transform=transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=32,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=32,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, valid_loader


def train_pcb():
    ARGS = DotDict(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(48, 64, 80, 80),
        strides=(2, 2),
        num_res_units=1,
        lr=1e-3,
    )
    NUM_EPOCH = 1000

    all_images = get_pcb_images()
    n = len(all_images)
    train_idx = np.arange(n)[:int(n * 0.8)]
    valid_idx = np.arange(n)[int(n * 0.8):]
    train_images = [{'image': all_images[i]} for i in train_idx]
    valid_images = [{'image': all_images[i]} for i in valid_idx]
    train_loader, valid_loader = get_pcb_loaders(train_images, valid_images)
    # x = next(iter(train_loader))  # check
    model = Model(ARGS)
    torch.set_float32_matmul_precision('medium')
    checkpoint = pl.callbacks.ModelCheckpoint(
        save_last=True,
        save_top_k=1, 
        monitor='val_loss', 
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", 
        mode="min", 
        patience=20
    )
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCH,
        accelerator="gpu",
        devices=[0],
        num_nodes=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=[
            checkpoint, 
            early_stopping, 
        ],
        default_root_dir="/root/autodl-tmp/kaggle/anomaly/jerry_anomaly"
    )
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    train_pcb()