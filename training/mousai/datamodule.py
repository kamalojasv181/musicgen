from math import pi 

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import load_dataset

import librosa

def load_audio(batch):

    batch["wave"] = []

    for idx, path in enumerate(batch["path"]):
            
        audio, sr = librosa.load(path, sr=48000, mono=False)
        audio = audio[:, :2097152]

        batch["wave"].append(audio)

    return batch

class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_folders: list,
        valid_folders: list,
        test_folders: list,
        batch_size: int,
        *,
        num_workers: int,
        num_proc: int = 16,
        **kwargs: int,
    ) -> None:
        super().__init__()

        train_files = [folder + "/data.json" for folder in train_folders]
        valid_files = [folder + "/data.json" for folder in valid_folders]
        test_files = [folder + "/data.json" for folder in test_folders]

        self.dataset_train = load_dataset("json", data_files=train_files, num_proc=num_proc).with_format("torch")["train"].map(load_audio, batched=True, num_proc=num_proc)

        self.dataset_valid = load_dataset("json", data_files=valid_files, num_proc=num_proc).with_format("torch")["train"].map(load_audio, batched=True, num_proc=num_proc)

        self.dataset_test = load_dataset("json", data_files=test_files, num_proc=num_proc).with_format("torch")["train"].map(load_audio, batched=True, num_proc=num_proc)

        self.num_workers = num_workers
        self.batch_size = batch_size

    def get_dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset=dataset,            
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
            shuffle=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_train)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_valid)
    
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_test)
