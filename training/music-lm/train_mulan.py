import torch
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from datasets import load_dataset, load_from_disk
import argparse
import copy
import sys
import wandb
import yaml
sys.path.append('../../')

from utils import *


def main(config):

    if config.wandb.mode == "offline":
        os.environ["WANDB_MODE"] = "offline"

    if not os.path.exists(config.wandb.path):
        os.makedirs(config.wandb.path)

    wandb.init(project=config.wandb.project, name = config.wandb.run_name, entity = os.environ["WANDB_ENTITY"], dir = config.wandb.path, job_type = config.wandb.job_type)

    with open(os.path.join(wandb.run.dir, "config_process.yaml"), "w") as f:
        yaml.dump(config.__dict__, f)

    # set seed
    set_seed(config.seed)

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

    model = MuLaN(
        audio_transformer = audio_transformer,
        text_transformer = text_transformer
    )

    # put model on device
    model = model.to(config.device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    dataset = load_from_disk(config.dataset_path)

    # make a dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=config.batch_size, shuffle=True)

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
    for epoch in range(config.max_epochs):

        if patience == 0 or total_iters > config.max_iters:
            break

        model.train()

        # train
        for batch_idx, batch in enumerate(train_dataloader):

            texts, audios = batch['text'], batch['audio']

            # put audios on device
            audios = audios.to(config.device)
            texts = texts.to(config.device)

            # squeeze audios and texts
            audios = audios.squeeze(1)
            texts = texts.squeeze(1)

            # do forward pass
            loss = model(wavs=audios, texts=texts)

            # do backward pass
            loss.backward()

            # update parameters
            optimizer.step()

            # zero out gradients
            optimizer.zero_grad()

            # print loss
            print(f'Running Training Loss: {loss.item()}')
            wandb.log({'Running Training Loss': loss.item()})

            # append loss to list
            training_loss_list.append(loss.item())

            # increment total iters
            total_iters += 1

            iters_before_eval -= 1

            if iters_before_eval == 0:

                print('Evaluating ...')

                # evaluate on test set
                model.eval()

                test_loss = 0

                for batch_idx, batch in enumerate(test_loader):

                    texts, audios = batch['text'], batch['audio']

                    # put audios on device
                    audios = audios.to(config.device)
                    texts = texts.to(config.device)

                    audios = audios.squeeze(1)
                    texts = texts.squeeze(1)

                    # do forward pass
                    loss = model(wavs=audios, texts=texts)

                    test_loss += loss.item()

                test_loss /= len(test_loader)

                print(f'Averaged Test Loss: {test_loss}')
                wandb.log({'Averaged Test Loss': test_loss})

                print(f'Averaged Training Loss: {sum(training_loss_list) / len(training_loss_list)}')
                wandb.log({'Averaged Training Loss': sum(training_loss_list) / len(training_loss_list)})

                # append test loss to list
                test_loss_list.append(test_loss)

                # reset training loss list
                training_loss_list = []

                # reset iters before eval
                iters_before_eval = config.iters_before_eval

                if test_loss < best_test_loss:
                        
                    best_test_loss = test_loss
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                    best_iter = total_iters

                    # save best model
                    torch.save(best_model, os.path.join(wandb.run.dir, 'best_model.pt'))

                # do early stopping
                if epoch > 1 and test_loss_list[-1] > test_loss_list[-2]:
                    patience -= 1

                    if patience == 0:
                        print('Early Stopping ...')
                        break

                else:
                    patience = config.patience


    print(f'Best Test Loss: {best_test_loss}')
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Iter: {best_iter}')

    table = wandb.Table(
        columns=["Test Loss", "Epoch", "Iteration"],
        data=[[best_test_loss, best_epoch, best_iter]]
    )

    wandb.log({"Best Model": table})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config' , type=str, default='config_mulan.yaml')

    args = parser.parse_args()

    config = load_config(args.config)

    main(config)
