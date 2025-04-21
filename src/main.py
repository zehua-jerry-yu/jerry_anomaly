from utils import *
from model import *
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import cv2
import timm

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


def get_pcb_images(normal):

    # load rect info from xml
    import xml.etree.ElementTree as ET
    tree = ET.parse('/root/autodl-tmp/kaggle/anomaly/pcb2/template/template.xml')
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

    from PIL import Image
    PATH_DATA = "/root/autodl-tmp/kaggle/anomaly/pcb2/aligned"
    filenames = os.listdir(PATH_DATA)
    filenames = [f for f in filenames if f.endswith(".JPG")]
    if normal:
        filenames = [f for f in filenames if len(f) == 8]  # only normal samples
    else:
        filenames = [f for f in filenames if len(f) == 7]  # only anomalous
    all_images = []

    template = cv2.imread('/root/autodl-tmp/kaggle/anomaly/pcb2/template/template2.JPG', cv2.IMREAD_GRAYSCALE)
    
    for filename in filenames:

        # template matching
        image = cv2.imread(os.path.join(PATH_DATA, filename), cv2.IMREAD_GRAYSCALE)
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # return idx in width, idx in height

        image = Image.open(os.path.join(PATH_DATA, filename))
        image = np.array(image)  # (H,W,C)
        image = np.moveaxis(image, 2, 0)  # (C,H,W)
        image = image[:, max_loc[1]: max_loc[1] + template.shape[0], max_loc[0]: max_loc[0] + template.shape[1]]
        # assert(image.shape[1] == height)
        # assert(image.shape[2] == width)
        # image = image[:, ymin: ymax + 1, xmin: xmax + 1]
        all_images.append(image)
    
    # save the matched images for visualization
    for filename, image in zip(filenames, all_images):
        from PIL import Image
        x = np.moveaxis(image, 0, 2)
        x = Image.fromarray(x, 'RGB')
        x.save(os.path.join("/root/autodl-tmp/kaggle/anomaly/pcb2/matched", f"{filename}.jpg"))
    
    return filenames, all_images
    

class GetMask(object):
    # transform. given image, generate random mask.
    def __init__(self, k=40, nfolds=5):
        self.k = k
        self.nfolds = nfolds

    def __call__(self, y):
        y = y['image']
        c, h, w = y.shape
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
        x = np.where(mask[np.newaxis], 0, y)
        return {'image': x, 'label': y, 'mask': mask}


def get_nonrandom_masks(h, w, k=40, nfolds=5):
    assert h % k == 0
    assert w % k == 0
    masks = []
    for t in range(nfolds):
        mask = np.full((h, w), False, dtype=bool)
        for i in range(h // k):
            for j in range(w // k):
                if (i * (w // k) + j) % nfolds == t:
                    mask[i * k: i * k + k, j * k: j * k + k] = True
        masks.append(torch.from_numpy(mask))
    return masks

    
def get_pcb_loaders(train_images, valid_images, use_mask):
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 64

    transforms = Compose([
        NormalizeIntensityd(keys="image"),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        Resized(spatial_size=(192, 224), size_mode='all', keys="image"),
    ])
    if use_mask:
        transforms = Compose([
            transforms,
            GetMask()
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
        strides=(2, 2, 1),
        num_res_units=1,
        lr=1e-3,
    )
    NUM_EPOCH = 100

    _, all_images = get_pcb_images(normal_only=True)
    n = len(all_images)
    train_idx = np.arange(n)#[:int(n * 0.8)]
    valid_idx = np.arange(n)[int(n * 0.8):]
    train_images = [{'image': all_images[i]} for i in train_idx]
    valid_images = [{'image': all_images[i]} for i in valid_idx]
    train_loader, valid_loader = get_pcb_loaders(train_images, valid_images)
    # sample = next(iter(train_loader))  # check
    model = Model(ARGS)
    torch.set_float32_matmul_precision('medium')
    checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,  # save all
        every_n_epochs=10,
        monitor='val_loss', 
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCH,
        accelerator="gpu",
        devices=[0],
        num_nodes=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=[
            checkpoint
        ],
        default_root_dir="/root/autodl-tmp/kaggle/anomaly/jerry_anomaly"
    )
    trainer.fit(model, train_loader, valid_loader)


def test_pcb():
    model_paths = "/root/autodl-tmp/kaggle/anomaly/jerry_anomaly/lightning_logs/version_3/checkpoints"
    out_path = "/root/autodl-tmp/kaggle/anomaly/pcb2_preds"
    for filename in os.listdir(out_path):
        os.remove(os.path.join(out_path, filename))  # clear folder first

    filenames, all_images = get_pcb_images(normal_only=False)
    filenames = filenames[:20]
    all_images = all_images[:20]

    h, w = (192, 224)
    masks = get_nonrandom_masks(h=h, w=w)
    transforms = Compose([
        NormalizeIntensityd(keys="image"),
        Resized(spatial_size=(h, w), size_mode='all', keys="image"),
    ])

    for model_path in os.listdir(model_paths):
        epoch = model_path.split('-')[0]
        model = Model.load_from_checkpoint(os.path.join(model_paths, model_path))
        model.eval()
        model.to("cuda")

        result = pd.Series()
        for filename, image in zip(filenames, all_images):
            print(filename)
            x = transforms({'image': image})  # add dim for batch
            with torch.no_grad():
                y_pred = torch.zeros_like(x['image'])
                for i, mask in enumerate(masks):
                    xx = x['image'].clone().detach()
                    xx = torch.where(mask[None], 0, xx)
                    xx = xx[None].to("cuda")
                    out = model(xx)
                    y_pred[:, mask] = out[0][:, mask].to("cpu")
                loss = torch.mean(torch.square(y_pred - x['image']))
                result[filename] = loss.item()
                # inv transform the predicted picture then save as jpg
                y_pred = y_pred.numpy()
                y_pred = y_pred * image.std() + image.mean()
                y_pred = y_pred.astype(int)
                y_pred = np.maximum(y_pred, 0)
                y_pred = np.minimum(y_pred, 255)
                y_pred = y_pred.astype(np.uint8)
                y_pred = np.moveaxis(y_pred, 0, 2)
                from PIL import Image
                img = Image.fromarray(y_pred, 'RGB')
                img.save(os.path.join(out_path, f"{filename.split('.')[0]}_{epoch}.jpg"))

    print(result)      
    import pdb; pdb.set_trace()


def train_pcb_simple_unsupervised():
    NUM_EPOCH = 10

    _, all_images = get_pcb_images(normal=True)
    n = len(all_images)
    train_idx = np.arange(n)[:int(n * 0.8)]
    valid_idx = np.arange(n)[int(n * 0.7): int(n * 0.8)]
    train_images = [{'image': all_images[i]} for i in train_idx]
    valid_images = [{'image': all_images[i]} for i in valid_idx]
    train_loader, valid_loader = get_pcb_loaders(train_images, valid_images, use_mask=False)
    model = ModelSimple()
    torch.set_float32_matmul_precision('medium')
    checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,  # save all
        every_n_epochs=2,
        monitor='val_loss', 
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCH,
        accelerator="gpu",
        devices=[0],
        num_nodes=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=[
            checkpoint
        ],
        default_root_dir="/root/autodl-tmp/kaggle/anomaly/jerry_anomaly"
    )
    trainer.fit(model, train_loader, valid_loader)


def test_pcb_simple_unsupervised():
    model_paths = "/root/autodl-tmp/kaggle/anomaly/jerry_anomaly/lightning_logs/version_0/checkpoints"
    out_path = "/root/autodl-tmp/kaggle/anomaly/pcb2_preds"
    for filename in os.listdir(out_path):
        os.remove(os.path.join(out_path, filename))  # clear folder first

    # add normal images that were not in training
    normal_filenames, all_images = get_pcb_images(normal=True)
    n = len(all_images)
    test_idx = np.arange(n)[int(n * 0.8):]
    normal_filenames = [normal_filenames[i] for i in test_idx]
    test_filenames = [i for i in normal_filenames]
    test_images = [all_images[i] for i in test_idx]
    
    # add all abnormal images
    anomaly_filenames, all_images = get_pcb_images(normal=False)
    test_filenames += anomaly_filenames
    test_images += all_images

    h, w = (192, 224)
    transforms = Compose([
        NormalizeIntensityd(keys="image"),
        Resized(spatial_size=(h, w), size_mode='all', keys="image"),
    ])

    for model_path in os.listdir(model_paths)[-1:]:
        epoch = model_path.split('-')[0]
        model = ModelSimple.load_from_checkpoint(os.path.join(model_paths, model_path))
        model.eval()
        model.to("cuda")

        result = pd.Series()
        for filename, image in zip(test_filenames, test_images):
            print(filename)
            x = transforms({'image': image})  # add dim for batch
            x = x['image'].to("cuda")
            with torch.no_grad():
                y_pred = model(x[None])[0]
                loss = torch.mean(torch.square(y_pred - x))
                result[filename] = loss.item()
                # inv transform the predicted picture then save as jpg
                y_pred = y_pred.cpu().numpy()
                y_pred = y_pred * image.std() + image.mean()
                y_pred = y_pred.astype(int)
                y_pred = np.maximum(y_pred, 0)
                y_pred = np.minimum(y_pred, 255)
                y_pred = y_pred.astype(np.uint8)
                y_pred = np.moveaxis(y_pred, 0, 2)
                from PIL import Image
                img = Image.fromarray(y_pred, 'RGB')
                img.save(os.path.join(out_path, f"{filename.split('.')[0]}_{epoch}.jpg"))

    result_normal = result.loc[normal_filenames]
    result_anomaly = result.loc[anomaly_filenames]
    result.sort_values(inplace=True)
    print(result)
    import pdb; pdb.set_trace()


def get_supervised_pcb_loaders(samples):
    BATCH_SIZE = 16

    transforms = Compose([
        NormalizeIntensityd(keys="image"),
        Resized(spatial_size=(192, 224), size_mode='all', keys="image"),
    ])
    ds = Dataset(data=samples, transform=transforms)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=32,
        pin_memory=True
    )
    return loader


def pcb_simple_supervised():
    model_name = "efficientnet_b0"  # Lightweight and good
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 → feature extractor
    model = model.to('cuda')
    model.eval()

    _, normal_images = get_pcb_images(normal=True)
    _, anomalous_images = get_pcb_images(normal=False)
    train_samples = []
    for image in normal_images:
        train_samples.append({
            'image': image,
            'label': False
        })
    for image in anomalous_images:
        train_samples.append({
            'image': image,
            'label': True
        })

    train_loader = get_supervised_pcb_loaders(train_samples)

    features = []
    labels = []
    with torch.no_grad():
        for batch in train_loader:
            imgs = batch['image']
            lbls = batch['label']
            imgs = imgs.to('cuda')
            feats = model(imgs)  # Shape: [batch, feature_dim]
            features.append(feats.cpu().numpy())
            labels.append(lbls.cpu().numpy())
    X = np.concatenate(features)
    y = np.concatenate(labels)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
        log_loss,
        brier_score_loss
    )
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

    all_true, all_pred, all_probs = [], [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold}/6 ===")
        # Extract features
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx], y[val_idx]

        # Train classifier
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X_train, y_train)

        # Predict on validation fold
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)

        # Accumulate
        all_true.extend(y_val)
        all_pred.extend(y_pred)
        all_probs.extend(y_prob[:, 1])

    print("\n=== Final Classification Report ===")
    print(classification_report(all_true, all_pred))

    print("=== Final Confusion Matrix ===")
    print(confusion_matrix(all_true, all_pred))
    print(f"ROC AUC:           {roc_auc_score(all_true, all_probs):.4f}")
    print(f"PR AUC:            {average_precision_score(all_true, all_probs):.4f}")
    print(f"Log Loss:          {log_loss(all_true, all_probs):.4f}")
    print(f"Brier Score:       {brier_score_loss(all_true, all_probs):.4f}")
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    # train_pcb()
    # test_pcb()

    # "simple" means input and output are same image
    # using unet for this would make it learn identical transformation in 1 epoch

    # train_pcb_simple_unsupervised()
    # test_pcb_simple_unsupervised()

    pcb_simple_supervised()