import torch
from torch import nn
from monai.networks.nets import UNet
import lightning.pytorch as pl


def masked_mse_loss(y_hat, y, mask):
    y = torch.moveaxis(y, 1, 0)
    y = y.reshape(y.shape[0], -1)
    y_hat = torch.moveaxis(y_hat, 1, 0)
    y_hat = y_hat.reshape(y_hat.shape[0], -1)
    mask = mask.flatten()
    return torch.mean(torch.square(y[:, mask] - y_hat[:, mask]))


class Model(pl.LightningModule):
    def __init__(self, args):
    
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=args.channels,
            strides=args.strides,
            num_res_units=args.num_res_units,
        )

        self.train_loss = 0
        self.valid_loss = 0
        self.num_train_batch = 0
        self.num_val_batch = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch['image'], batch['label'], batch['mask']
        y_hat = self(x)
        loss = masked_mse_loss(y_hat, y, mask)
        self.train_loss += loss
        self.num_train_batch += 1
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        loss_per_epoch = self.train_loss/self.num_train_batch
        print(f"Epoch {self.current_epoch} - Average Train Loss: {loss_per_epoch:.4f}")
        self.log('train_loss', loss_per_epoch, prog_bar=False)
        self.train_loss = 0
        self.num_train_batch = 0
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad(): # This ensures that gradients are not stored in memory
            x, y, mask = batch['image'], batch['label'], batch['mask']
            y_hat = self(x)
            loss = masked_mse_loss(y_hat, y, mask)
            self.valid_loss += loss
            self.num_val_batch += 1
        torch.cuda.empty_cache()
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        loss_per_epoch = self.valid_loss/self.num_val_batch
        print(f"Epoch {self.current_epoch} - Average Val Loss: {loss_per_epoch:.4f}")
        self.log('val_loss', loss_per_epoch, prog_bar=False, sync_dist=False)
        self.valid_loss = 0
        self.num_val_batch = 0
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.args.lr)


class LightConvAutoencoder(nn.Module):
    def __init__(self):
        super(LightConvAutoencoder, self).__init__()
        
        self.IMAGE_SHAPE = (3, 192, 224)

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(32 * self.IMAGE_SHAPE[1] * self.IMAGE_SHAPE[2] // 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # bottleneck
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * self.IMAGE_SHAPE[1] * self.IMAGE_SHAPE[2] // 16),
            nn.ReLU()
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # Encoder
        x = self.encoder_conv(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.encoder_fc(x)

        # Decoder
        x = self.decoder_fc(x)
        x = x.view(batch_size, 32, self.IMAGE_SHAPE[1] // 4, self.IMAGE_SHAPE[2] // 4)
        x = self.decoder_conv(x)
        return x


class ModelSimple(pl.LightningModule):
    def __init__(self):
    
        super().__init__()
        self.save_hyperparameters()
        self.model = LightConvAutoencoder()

        self.train_loss = 0
        self.valid_loss = 0
        self.num_train_batch = 0
        self.num_val_batch = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = self(x)
        loss = torch.mean(torch.square(x - y))
        self.train_loss += loss
        self.num_train_batch += 1
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        loss_per_epoch = self.train_loss/self.num_train_batch
        print(f"Epoch {self.current_epoch} - Average Train Loss: {loss_per_epoch:.4f}")
        self.log('train_loss', loss_per_epoch, prog_bar=False)
        self.train_loss = 0
        self.num_train_batch = 0
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad(): # This ensures that gradients are not stored in memory
            x = batch['image']
            y = self(x)
            loss = torch.mean(torch.square(x - y))
            self.valid_loss += loss
            self.num_val_batch += 1
        torch.cuda.empty_cache()
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        loss_per_epoch = self.valid_loss/self.num_val_batch
        print(f"Epoch {self.current_epoch} - Average Val Loss: {loss_per_epoch:.4f}")
        self.log('val_loss', loss_per_epoch, prog_bar=False, sync_dist=False)
        self.valid_loss = 0
        self.num_val_batch = 0
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)