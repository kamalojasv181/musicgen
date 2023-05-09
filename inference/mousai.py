import argparse
from transformers import AutoModel
from utils import *
import yaml
import os
import torch
import sys

sys.path.append("../")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='mousai.yaml')
    parser.add_argument("--text_prompts", type=str, default='../prompts/electronic.txt')
    parser.add_argument("--output_dir", type=str, default='../output')
    args = parser.parse_args()

    # load the config
    config = load_config(args.config_path)

    # set seed
    set_seed(config.seed)

    # get the device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    

    # load the model
    model = AutoModel.from_pretrained(config.model_name, trust_remote_code=True, use_auth_token=access_token).to(device)

    # load the prompts
    prompts = read_txt(args.text_prompts).split('\n')

    # generate the audios
    samples, info = model.sample(
    text=prompts,
    sampling_steps=100,
    decoding_steps=100, 
    cfg_scale=8.0, 
    seed=42, 
    length=2048
    )

    # save the audios
    for i, (sample, prompt) in enumerate(zip(samples, info["text"])):
        sample_cpu = sample.cpu().numpy()

        # if the output dir does not exist, create it
        if not os.path.exists(config.output_dir):
            os.mkdir(config.output_dir)

        save_file_name =  song_prompt_to_name(prompt)

        # save the audio
        save_wav(sample_cpu, os.path.join(config.output_dir, save_file_name + ".wav"), config.sr)
 



