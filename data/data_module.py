import pytorch_lightning as pl

from torch.utils.data import DataLoader

from .dataset import TextAudioDataset
from .collater import TextAudioCollate


class TextAudioDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.collate_fn = TextAudioCollate()

    def setup(self, stage=None):
        self.train_ds = TextAudioDataset(self.params.data.train_file, self.params.data)
        self.valid_ds = TextAudioDataset(self.params.data.valid_file, self.params.data)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.params.train.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.params.train.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=8
        )
