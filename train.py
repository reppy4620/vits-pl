import warnings

warnings.filterwarnings('ignore')

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import load_config
from data import TextAudioDataModule
from models import VITSModule


def main():
    config = load_config('./configs/base.yaml')

    pl.seed_everything(config.seed)

    model = VITSModule(config)
    dm = TextAudioDataModule(config)

    mc = ModelCheckpoint(
        filename='VITS_{epoch: 06d}',
        save_last=True,
        every_n_val_epochs=100
    )

    trainer = pl.Trainer(
        callbacks=[mc],
        gpus=torch.cuda.device_count(),
        **config.trainer
    )
    model.trainer = trainer
    trainer.fit(model=model, datamodule=dm)


if __name__ == '__main__':
    main()
