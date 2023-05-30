import random
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Type, Tuple 
from math import pi 

import json 
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
import torch.nn.functional as F 
from einops import rearrange, repeat
from ema_pytorch import EMA
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm 

from archisound import ArchiSound
from transformers import AutoModel 

""" Model """

torch.set_float32_matmul_precision('high')

from audio_diffusion_pytorch import UNetV0

UNetT = lambda: UNetV0

def dropout(proba: float):
    return random.random() < proba

class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_eps: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        model: nn.Module,
        embedding_mask_proba: float,
        autoencoder_name: str 
    ):
        super().__init__()
        self.lr = lr
        self.lr_eps = lr_eps
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_weight_decay = lr_weight_decay
        self.model = model
        self.model_ema = EMA(self.model, beta=ema_beta, power=ema_power)
        self.embedding_mask_proba = embedding_mask_proba
        self.autoencoder = AutoModel.from_pretrained(
            f"archinetai/{autoencoder_name}", trust_remote_code=True
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def get_texts(self, infos: List[Dict], use_dropout: bool = True) -> List[str]:
        texts = []
        for info in infos:
            tags = [] 
            if 'title' in info and not (dropout(0.1) and use_dropout):
                tags.append(info['title'][0])
            if 'genre' in info and not (dropout(0.1) and use_dropout):
                tags.append(info['genre'][0])
            if 'artist' in info and not (dropout(0.1) and use_dropout):
                tags.append(info['artist'][0])
            if 'album' in info and not (dropout(0.2) and use_dropout):
                tags.append(info['album'][0])
            if 'year' in info and not (dropout(0.2) and use_dropout):
                tags.append(info['year'][0])
            if ('crop_id' and 'num_crops' in info) and not (dropout(0.2) and use_dropout): 
                tags.append(f"{info['crop_id']+1} of {info['num_crops']}")
            random.shuffle(tags)
            text = ' '.join(tags) if random.random() > 0.5 else ', '.join(tags)
            texts.append(text)
        return texts

    @torch.no_grad() 
    def encode(self, wave: Tensor) -> Tensor:
        return self.autoencoder.encode(wave)

    def training_step(self, batch, batch_idx):
        wave, info = batch
        latent = self.encode(wave)
        text = self.get_texts(info) 
        loss = self.model(latent, text=text, embedding_mask_proba=self.embedding_mask_proba)
        self.log("train_loss", loss, sync_dist=True)
        self.model_ema.update()
        self.log("ema_decay", self.model_ema.get_current_decay(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wave, info = batch 
        latent = self.encode(wave)
        text = self.get_texts(info) 
        loss = self.model(latent, text=text)
        self.log("valid_loss", loss, sync_dist=True)
        return loss


""" Datamodule """

class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_train,
        dataset_valid,
        *,
        num_workers: int,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.num_workers = num_workers

    def get_dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset=dataset,            
            num_workers=self.num_workers,
            batch_size=None, 
            shuffle=False,
            pin_memory=True,
            prefetch_factor=2,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_train)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_valid)


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    print("WandbLogger not found.")
    return None


def log_wandb_audio_batch(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{id}": wandb.Audio(
                samples[idx],
                caption=caption,
                sample_rate=sampling_rate,
            )
            for idx in range(num_items)
        }
    )


def log_wandb_audio_spectrogram(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = samples.detach().cpu()
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        center=True,
        norm="slaney",
    )

    def get_spectrogram_image(x):
        spectrogram = transform(x[0])
        image = torchaudio.functional.amplitude_to_DB(spectrogram, 1.0, 1e-10, 80.0)
        trace = [go.Heatmap(z=image, colorscale="viridis")]
        layout = go.Layout(
            yaxis=dict(title="Mel Bin (Log Frequency)"),
            xaxis=dict(title="Frame"),
            title_font_size=10,
        )
        fig = go.Figure(data=trace, layout=layout)
        return fig

    logger.log(
        {
            f"mel_spectrogram_{idx}_{id}": get_spectrogram_image(samples[idx])
            for idx in range(num_items)
        }
    )


def log_wandb_embeddings(logger: WandbLogger, id: str, embeddings: Tensor):
    num_items = embeddings.shape[0]
    embeddings = embeddings.detach().cpu()

    def get_figure(x):
        trace = [go.Heatmap(z=x, colorscale="viridis")]
        fig = go.Figure(data=trace)
        return fig

    logger.log(
        {
            f"embedding_{idx}_{id}": get_figure(embeddings[idx])
            for idx in range(num_items)
        }
    )

def log_wandb_texts(logger: WandbLogger, texts: List[str], id: str = ''):
    def get_figure(texts):
        return go.Figure(data=[go.Table(
            #header=dict(values=[f'Texts']), 
            cells=dict(values=[texts]))
        ])
    logger.log(
        {
            f"texts_{id}": get_figure(texts)
        }
    )


class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        sampling_steps: List[int],
        decoder_sampling_steps: int, 
        use_ema_model: bool,
        embedding_scale: int, 
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.sampling_steps = sampling_steps
        self.decoder_sampling_steps = decoder_sampling_steps
        self.embedding_scale = embedding_scale 
        self.use_ema_model = use_ema_model
        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        model = pl_module.model

        if self.use_ema_model:
            model = pl_module.model_ema.ema_model

        waveform, info = batch
        waveform = waveform[0 : self.num_items]
        info = info[0 : self.num_items]

        log_wandb_audio_batch(
            logger=wandb_logger,
            id="true",
            samples=waveform,
            sampling_rate=self.sampling_rate, 
        )
        log_wandb_audio_spectrogram(
            logger=wandb_logger,
            id="true",
            samples=waveform,
            sampling_rate=self.sampling_rate,
        )

        # Log texts 
        text = pl_module.get_texts(info)
        log_wandb_texts(logger=wandb_logger, id='sample', texts=text)
        noise = torch.randn_like(pl_module.autoencoder.encode(waveform))

        for steps in self.sampling_steps:
            latent_samples = model.sample(
                noise, 
                num_steps=steps,
                text=text,
                embedding_scale=self.embedding_scale, 
            )
            samples = pl_module.autoencoder.decode(
                latent_samples, 
                num_steps=self.decoder_sampling_steps
            )
            log_wandb_audio_batch(
                logger=wandb_logger,
                id="sample",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps",
            )
            log_wandb_audio_spectrogram(
                logger=wandb_logger,
                id="sample",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps",
            )

        if is_train:
            pl_module.train()