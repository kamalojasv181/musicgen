import os
import glob
import mutagen
from pydub import AudioSegment
import random
import argparse
import pandas as pd
from datasets import Dataset
import json
from tqdm import tqdm

def get_metadata(audio_file):
    audio = mutagen.File(audio_file)

    if audio is None:
        return None
    
    metadata = {
        "title": [audio.tags["TIT2"].text[0]] if "TIT2" in audio.tags else ["__None__"],
        "artist": [audio.tags["TPE1"].text[0]] if "TPE1" in audio.tags else ["__None__"],
        "album": [audio.tags["TALB"].text[0]] if "TALB" in audio.tags else ["__None__"],
        "year": [str(audio.tags["TDRC"].text[0])[0:4]] if "TDRC" in audio.tags else ["__None__"],
        "genre": [audio.tags["TCON"].text[0]] if "TCON" in audio.tags else ["__None__"],
    }

    return metadata

def check(datapoint, wave):

    if wave is None:
        return False
    
    if datapoint["info"] is None:
        return False
    
    if datapoint["info"]["title"][0] is None and datapoint["info"]["artist"][0] is None and datapoint["info"]["album"][0] is None and datapoint["info"]["genre"][0] is None and datapoint["info"]["year"][0] is None and datapoint["info"]["crop_id"] is None and datapoint["info"]["num_crops"] is None:

        return False
    
    return True


if __name__ == '__main__':

    # create an argument parser

    parser = argparse.ArgumentParser(description="Prepare the dataset for training")

    # add arguments to the parser

    parser.add_argument("--data_folder", type=str, help="Path to the folder containing the audio files", default="../data/sample_dataset")
    parser.add_argument("--output_folder", type=str, help="Path to the folder where the processed dataset will be stored", default="../data/processed_dataset")
    parser.add_argument("--continue_index", type=int, help="If we want to continue from an index", default=0)
    args = parser.parse_args()

    # create the output folder if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    data_folder = args.data_folder

    audio_files = glob.glob(os.path.join(data_folder, "*.mp3"))

    for idx, audio_file in tqdm(enumerate(audio_files)):

        if idx < args.continue_index:
            continue

        metadata = get_metadata(audio_file)

        # get the audio as dual channel
        audio = AudioSegment.from_mp3(audio_file)

        if audio.channels != 2:
            continue
        
        # find the length of the audio in milliseconds
        audio_len = len(audio)

        if audio_len < 44000:
            continue

        # get 4 random chunks of 44000 samples each
        starts = []

        for i in range(4):
            starts.append(random.randint(0, audio_len - 44000))

        # sort the starts

        starts.sort()

        # get the chunks

        for i in range(4):
            audio_chunk = audio[starts[i]:starts[i] + 44000]

            datapoint_new = {
                "path": f"{args.output_folder}/{audio_file.split('/')[-1][:-4]}_{i}.mp3",
                "info": metadata,
            }

            datapoint_new["info"]["crop_id"] = i
            datapoint_new["info"]["num_crops"] = 4

            if not check(datapoint_new, audio_chunk):
                continue

            # save the audio
            audio_chunk.export(f"{args.output_folder}/{audio_file.split('/')[-1][:-4]}_{i}.mp3", format="mp3")
            
            with open(f"{args.output_folder}/data.json", "a") as f:
                f.write(json.dumps(datapoint_new))
                f.write("\n")