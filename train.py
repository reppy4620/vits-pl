import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

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
        verbose=True,
        save_last=True,
        period=25
    )

    trainer = pl.Trainer(
        callbacks=[mc],
        **config.trainer
    )
    model.trainer = trainer
    trainer.fit(model=model, datamodule=dm)


if __name__ == '__main__':
    main()
