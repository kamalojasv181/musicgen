import os
import glob
import mutagen
import librosa
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

def check(datapoint):

    # # filter out the points in train such that wave is not none and shape of wave is (2, 2097152). Also, remove the points where "info" is None or all of info["title"][0] = None, info["artist"][0] = None, info["album"][0] = None, info["genre"][0] = None, info["year"][0] = None, info["crop_id"] = None and info["num_crops"] = None

    if datapoint["wave"] is None:
        return False
    
    if datapoint["info"] is None:
        return False
    
    if len(datapoint["wave"]) != 2:
        return False
    
    if len(datapoint["wave"][0]) != 2097152 and len(datapoint["wave"][1]) != 2097152:
        return False
    
    if datapoint["info"]["title"][0] is None and datapoint["info"]["artist"][0] is None and datapoint["info"]["album"][0] is None and datapoint["info"]["genre"][0] is None and datapoint["info"]["year"][0] is None and datapoint["info"]["crop_id"] is None and datapoint["info"]["num_crops"] is None:

        return False
    
    return True


if __name__ == '__main__':

    # create an argument parser

    parser = argparse.ArgumentParser(description="Prepare the dataset for training")

    # add arguments to the parser

    parser.add_argument("--data_folder", type=str, help="Path to the folder containing the audio files")

    args = parser.parse_args()

    data_folder = args.data_folder

    audio_files = glob.glob(os.path.join(data_folder, "*.mp3"))

    for audio_file in tqdm(audio_files):

        metadata = get_metadata(audio_file)

        # get the audio as dual channel
        audio, sr = librosa.load(audio_file, sr=48000, mono=False)
        
        # find the length of the audio in number of samples
        try:
            audio_len = audio.shape[1]
        except:
            continue

        if audio_len < 2097152:
            continue

        # get 4 random chunks of 2097152 samples each
        starts = []

        for i in range(4):
            starts.append(random.randint(0, audio_len - 2097152))

        # sort the starts

        starts.sort()

        # get the chunks

        for i in range(4):
            audio_chunk = audio[:, starts[i]:starts[i] + 2097152]

            # convert chunk to list
            audio_chunk = audio_chunk.tolist()

            datapoint_new = {
                "wave": audio_chunk,
                "info": metadata,
            }

            datapoint_new["info"]["crop_id"] = i
            datapoint_new["info"]["num_crops"] = 4

            if not check(datapoint_new):
                continue
            
            with open(f"{data_folder}/train_data.json", "a") as f:
                f.write(json.dumps(datapoint_new))
                f.write("\n")
