import pytorch_lightning as pl
from omegaconf import OmegaConf
from module import Module
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from module import Module
from datamodule import Datamodule
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, RichModelSummary
from logger import SampleLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
import os
import shutil
import glob

if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")

    if not os.path.exists(config.validation.path):
        os.makedirs(config.validation.path)

        # make a folder called true
        os.makedirs(os.path.join(config.validation.path, "true"))

        # make a folder called generated
        os.makedirs(os.path.join(config.validation.path, "generated"))

        # also create an empty json file called prompt to path
        with open(os.path.join(config.validation.path, "prompt_to_path.json"), "w") as f:
            f.write("{}")

        for validation_folder in config.datamodule.dataset_valid.folders:
            # copy the audio files to the true folder
            # get the names of all mp3 files in the folder
            audio_files = glob.glob(os.path.join(validation_folder, "*.mp3"))

            for audio_file in audio_files:
                shutil.copy(audio_file, os.path.join(config.validation.path, "true"))


    model = DiffusionModel(
        net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
        in_channels=config.module.model.in_channels,
        dim=config.module.model.dim,
        channels=config.module.model.channels,
        factors=config.module.model.factors,
        items=config.module.model["items"],
        attentions=config.module.model.attentions,
        cross_attentions=config.module.model.cross_attentions,
        attention_heads=config.module.model.attention_heads,
        attention_features=config.module.model.attention_features,
        embedding_max_length=config.module.model.embedding_max_length,
        embedding_features=config.module.model.embedding_features,
        use_text_conditioning=config.module.model.use_text_conditioning,
        use_embedding_cfg=config.module.model.use_embedding_cfg,
        diffusion_t=VDiffusion, # The diffusion method used
        sampler_t=VSampler, # The diffusion sampler used
    )

    module = Module(
        lr=config.module.lr,
        lr_eps=config.module.lr_eps,
        lr_beta1=config.module.lr_beta1,
        lr_beta2=config.module.lr_beta2,
        lr_weight_decay=config.module.lr_weight_decay,
        ema_beta=config.module.ema_beta,
        ema_power=config.module.ema_power,
        model=model,
        embedding_mask_proba=config.module.embedding_mask_proba,
        autoencoder_name=config.module.autoencoder_name,
        validation_path=config.validation.path,
    )

    datamodule = Datamodule(
        train_folders=config.datamodule.dataset_train.folders,
        valid_folders=config.datamodule.dataset_valid.folders,
        test_folders=config.datamodule.dataset_test.folders,
        batch_size=config.datamodule.dataset_train.batch_size,
        num_workers=config.datamodule.num_workers,
        num_proc=config.datamodule.num_proc,
    )


    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=config.callbacks.model_checkpoint.dirpath,
            save_last=config.callbacks.model_checkpoint.save_last,
            every_n_train_steps=config.callbacks.model_checkpoint.every_n_train_steps,
        ),
        RichModelSummary(max_depth=config.callbacks.model_summary.max_depth),
        SampleLogger(
            num_items=config.callbacks.audio_samples_logger.num_items,
            channels=config.callbacks.audio_samples_logger.channels,
            sampling_rate=config.callbacks.audio_samples_logger.sampling_rate,
            sampling_steps=config.callbacks.audio_samples_logger.sampling_steps,
            decoder_sampling_steps=config.callbacks.audio_samples_logger.decoder_sampling_steps,
            embedding_scale=config.callbacks.audio_samples_logger.embedding_scale,
            use_ema_model=config.callbacks.audio_samples_logger.use_ema_model,
        ),
    ]

    logger = WandbLogger(
        project=config.loggers.wandb.project,
        entity=config.loggers.wandb.entity,
        job_type=config.loggers.wandb.job_type,
        group=config.loggers.wandb.group,
        save_dir=config.loggers.wandb.save_dir,
    )

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
#        devices=config.trainer.devices,
#        strategy=config.trainer.strategy,
#        num_nodes=config.trainer.num_nodes,
        precision=config.trainer.precision,
        accelerator=config.trainer.accelerator,
        min_epochs=config.trainer.min_epochs,
        max_epochs=config.trainer.max_epochs,
        enable_model_summary=config.trainer.enable_model_summary,
        log_every_n_steps=config.trainer.log_every_n_steps,
        limit_val_batches=config.trainer.limit_val_batches,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        val_check_interval=config.trainer.val_check_interval,
    )

    trainer.fit(module, datamodule=datamodule)
