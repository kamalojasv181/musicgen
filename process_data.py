import os
import json
import argparse
from datasets import load_dataset
import torch

def make_all_same_length(batch, max_length=96000):
    batch["audio"]["array"] = batch["audio"]["array"][:max_length]
    return batch

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--test_size', type=float, default=0.25, help='test size')
    parser.add_argument('--output_dir', type=str, default='data_hf', help='processed huggingface data directory')
    parser.add_argument('--max_length', type=int, default=96000, help='max length of audio')


    args = parser.parse_args()

    data_dir = args.data_dir

    files = os.listdir(data_dir)

    # make a directore called data inside data_dir
    os.makedirs(os.path.join(data_dir, 'data'), exist_ok=True)

    # put all the files in data_dir/data
    for file in files:
        os.rename(f'{data_dir}/{file}', f'{data_dir}/data/{file}')

    files = [f'data/{file}' for file in files]


    with open(f'{data_dir}/metadata.jsonl', 'w') as f:
        for file in files:
            json.dump({'file_name': file, 'name': file[5:-4]}, f)
            f.write('\n')

    data = load_dataset('audiofolder', data_dir=data_dir)

    data = data.map(make_all_same_length, args.max_length)

    data = data["train"].train_test_split(test_size=0.25)

    data = data.with_format("torch")

    data.save_to_disk(args.output_dir)