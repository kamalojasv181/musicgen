import torch
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from datasets import load_dataset
import argparse
import sys
sys.path.append('../../')

from utils import *


def main(config, args):

    # create audio_transformer object
    audio_transformer = AudioSpectrogramTransformer(
        dim=config.audio_transformer.dim,
        depth=config.audio_transformer.depth,
        heads=config.audio_transformer.heads,
        dim_head=config.audio_transformer.dim_head,
        spec_n_fft=config.audio_transformer.spec_n_fft,
        spec_win_length=config.audio_transformer.spec_win_length,
        spec_aug_stretch_factor=config.audio_transformer.spec_aug_stretch_factor
    )

    # create text_transformer object
    text_transformer = TextTransformer(
        dim=config.text_transformer.dim,
        depth=config.text_transformer.depth,
        heads=config.text_transformer.heads,
        dim_head=config.text_transformer.dim_head
    )

    mulan = MuLaN(
        audio_transformer = audio_transformer,
        text_transformer = text_transformer
    )

    # put mulan on device
    mulan = mulan.to(args.device)

    # create optimizer
    optimizer = torch.optim.Adam(mulan.parameters(), lr=config.lr)

    # get a ton of <sound, text> pairs and train
    dataset = load_dataset("args.text_audio_pairs")

    # make a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # train
    for batch_idx, batch in enumerate(dataloader):

        texts, audios = batch['text'], batch['audio']

        # put audios on device
        audios = audios.to(args.device)

        # do forward pass
        loss = mulan(wavs=audios, raw_texts=texts)

        # do backward pass
        loss.backward()

        # update parameters
        optimizer.step()

        # zero out gradients
        optimizer.zero_grad()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--text_audio_pairs', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--config' , type=str, default='config_mulan.yaml')

    args = parser.parse_args()

    config = load_config(args.config)

    main(config, args)
