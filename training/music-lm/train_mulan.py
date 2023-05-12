import torch
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from datasets import load_dataset
import argparse
import copy
import sys
sys.path.append('../../')

from utils import *


def main(config):

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
    mulan = mulan.to(config.device)

    # create optimizer
    optimizer = torch.optim.Adam(mulan.parameters(), lr=config.lr)

    # get a ton of <sound, text> pairs and train
    train_dataset = load_dataset("config.text_audio_pairs_train")

    # test dataset
    test_dataset = load_dataset("config.text_audio_pairs_test")

    # make a dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    best_test_loss = float('inf')
    best_model = None
    best_epoch = 0
    best_iter = 0

    patience = config.patience

    iters_before_eval = config.iters_before_eval

    test_loss_list = []
    training_loss_list = []

    total_iters = 0

    print(' Starting Training ....')
    for epoch in range(config.epochs):

        if patience == 0 or total_iters > config.max_iters:
            break

        model.train()

        # train
        for batch_idx, batch in enumerate(dataloader):

            texts, audios = batch['text'], batch['audio']

            # put audios on device
            audios = audios.to(config.device)

            # do forward pass
            loss = mulan(wavs=audios, raw_texts=texts)

            # do backward pass
            loss.backward()

            # update parameters
            optimizer.step()

            # zero out gradients
            optimizer.zero_grad()

            # print loss
            print(f'Running Training Loss: {loss.item()}')

            # append loss to list
            training_loss_list.append(loss.item())

            # increment total iters
            total_iters += 1

            iters_before_eval -= 1

            if iters_before_eval == 0:

                # evaluate on test set
                model.eval()

                test_loss = 0

                for batch_idx, batch in enumerate(test_loader):

                    texts, audios = batch['text'], batch['audio']

                    # put audios on device
                    audios = audios.to(config.device)

                    # do forward pass
                    loss = mulan(wavs=audios, raw_texts=texts)

                    test_loss += loss.item()

                test_loss /= len(test_loader)

                print(f'Averaged Test Loss: {test_loss}')
                print(f'Averaged Training Loss: {sum(training_loss_list) / len(training_loss_list)}')

                # append test loss to list
                test_loss_list.append(test_loss)

                # reset training loss list
                training_loss_list = []

                # reset iters before eval
                iters_before_eval = config.iters_before_eval

                if test_loss < best_test_loss:
                        
                    best_test_loss = test_loss
                    best_model = copy.deepcopy(mulan)
                    best_epoch = epoch
                    best_iter = total_iters

                    # save best model
                    torch.save(best_model, config.save_path)

                # do early stopping
                if epoch > 1 and test_loss_list[-1] > test_loss_list[-2]:
                    patience -= 1

                    if patience == 0:
                        print('Early Stopping ...')
                        break

                else:
                    patience = config.patience


    print(f'Best Test Loss: {best_test_loss}')
                



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--text_audio_pairs_train', type=str, required=True)
    parser.add_argument('--text_audio_pairs_test', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--config' , type=str, default='config_mulan.yaml')

    args = parser.parse_args()

    config = load_config(args.config)

    main(config, args)
