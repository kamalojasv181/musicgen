from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Type, Tuple 
from math import pi 

import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from einops import rearrange
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader


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
        self, trainer, pl_module, batch, batch_idx
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

        waveform, info = batch["wave"], batch["info"]
        waveform = waveform[0 : self.num_items]

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
        text = pl_module.get_texts(info)[0 : self.num_items]
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