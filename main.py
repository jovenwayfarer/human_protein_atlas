import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.metrics import Accuracy, Recall
from argparse import ArgumentParser
from tqdm import tqdm
from base_model import Net
from dataset import CellDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
import cv2


class HPALit(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        
        self.hparams = params
        self.lr = self.hparams.lr
        self.save_hyperparameters()
        self.model = Net(name = self.hparams.model)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_accuracy = Accuracy(subset_accuracy=True)
        self.val_accuracy = Accuracy(subset_accuracy=True)
        self.train_recall = Recall()
        self.val_recall = Recall()

        self.train_df = pd.read_csv(f'data_preprocessing/train_fold_{self.hparams.fold}.csv')
        self.valid_df = pd.read_csv(f'data_preprocessing/valid_fold_{self.hparams.fold}.csv')


        self.train_transforms = A.Compose([
                        A.Rotate(),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.Resize(width=self.hparams.img_size, height=self.hparams.img_size),
                        A.Normalize(),
                        ToTensorV2(),
                                        ])

        self.valid_transforms = A.Compose([
                        A.Resize(width=self.hparams.img_size, height=self.hparams.img_size),
                        A.Normalize(),
                        ToTensorV2(),
                                        ])


        self.train_dataset = CellDataset(data_dir=self.hparams.data_dir,
                                        csv_file=self.train_df, transform=self.train_transforms)
        self.val_dataset = CellDataset(data_dir=self.hparams.data_dir,
                                        csv_file=self.valid_df, transform=self.valid_transforms)

        

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.n_workers,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.n_workers,
                          pin_memory=True,
                          )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, eps=1e-6)
        lr_scheduler = {'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'valid_loss_epoch'}

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)

        train_loss = self.criterion(pred, y)
        self.train_accuracy(torch.sigmoid(pred), y.type(torch.int))
        self.train_recall(torch.sigmoid(pred), y.type(torch.int))

        
        return {'loss': train_loss}

    def training_epoch_end(self, outputs):
        train_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', train_loss_epoch)
        self.log('train_acc_epoch', self.train_accuracy.compute())
        self.log('train_recall_epoch', self.train_recall.compute())
        self.train_accuracy.reset()
        self.train_recall.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)

        val_loss = self.criterion(pred, y)
        self.val_accuracy(torch.sigmoid(pred), y.type(torch.int))
        self.val_recall(torch.sigmoid(pred), y.type(torch.int))
        

        return {'valid_loss': val_loss}


    def validation_epoch_end(self, outputs):
        val_loss_epoch = torch.stack([x['valid_loss'] for x in outputs]).mean()
        self.log('valid_loss_epoch', val_loss_epoch)
        self.log('valid_acc_epoch', self.val_accuracy.compute())
        self.log('valid_recall_epoch', self.val_recall.compute())
        self.val_accuracy.reset()
        self.val_recall.reset()



class MyPrintingCallback(Callback):

    def on_validation_epoch_end(self, trainer, pl_module):
    
        print('\n')
        print('Train Loss: {:.3f}'.format(trainer.callback_metrics['train_loss_epoch'].item()))
        print('Val Loss: {:.3f}'.format(trainer.callback_metrics['valid_loss_epoch'].item()))
        print('Train Accuracy: {:.3f}'.format(trainer.callback_metrics['train_acc_epoch'].item()))
        print('Val Accuracy: {:.3f}'.format(trainer.callback_metrics['valid_acc_epoch'].item()))
        print('Train Recall: {:.3f}'.format(trainer.callback_metrics['train_recall_epoch'].item()))
        print('Val Recall: {:.3f}'.format(trainer.callback_metrics['valid_recall_epoch'].item()))


class LitProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


def main(params):
    exp_name = f'{params.model}-{params.img_size}-fold-{params.fold}'
    checkpoint = ModelCheckpoint(
        dirpath=f'logs/{exp_name}',
        filename='{epoch}-{valid_loss_epoch:.3f}',
        save_top_k=-1,
        verbose=False,
    )
    printer = MyPrintingCallback()
    logger = CSVLogger(save_dir=f"logs/{exp_name}", name="text_logs")
    wandb_logger = WandbLogger(name=exp_name, project='all-data')
    bar = LitProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = HPALit(params)
    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        max_epochs=params.epochs,
        callbacks=[checkpoint, printer, bar, lr_monitor],
        logger=[logger, wandb_logger],
        gpus=1,
        num_sanity_val_steps=0,
        auto_lr_find=True,
    )
    trainer.tune(model)
    trainer.fit(model)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model', type=str, default='efficientnet_b1')

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--n_workers', type=int, default=24)
    parser.add_argument('--data_dir', type=str, required = True, help='dir of data to train on (cell-tiles)')
    return parser.parse_args()


if __name__ == '__main__':
    seed_everything(144)
    args = get_args()
    main(args)