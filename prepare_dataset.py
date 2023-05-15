import os
import mutagen
import random
import librosa
import argparse
from utils import *


def dropout(probability):
    return random.random() < probability

def save_wav(data, path, sr):
    import soundfile as sf
    sf.write(path, data, sr, "PCM_24")

def get_metadata(audio_file):
    audio = mutagen.File(audio_folder + "/" + audio_file)

    if audio is None:
        return None
    
    metadata = {
        "title": audio.tags["TIT2"].text[0] if "TIT2" in audio.tags else None,
        "artist": audio.tags["TPE1"].text[0] if "TPE1" in audio.tags else None,
        "album": audio.tags["TALB"].text[0] if "TALB" in audio.tags else None,
        "year": str(audio.tags["TDRC"].text[0])[0:4] if "TDRC" in audio.tags else None,
        "genre": audio.tags["TCON"].text[0] if "TCON" in audio.tags else None, 
        "path": audio_folder + "/" + audio_file,
        "length": audio.info.length,
    }

    return metadata

def get_text(metadata, use_dropout=True):

    text = []

    if metadata["title"] is not None and not (dropout(0.1) and use_dropout):
        text.append(metadata["title"])

    if metadata["genre"] is not None and not (dropout(0.1) and use_dropout):
        text.append(metadata["genre"])

    if metadata["artist"] is not None and not (dropout(0.1) and use_dropout):
        text.append(metadata["artist"])

    if metadata["album"] is not None and not (dropout(0.2) and use_dropout):
        text.append(metadata["album"])

    if metadata["year"] is not None and not (dropout(0.2) and use_dropout):
        text.append(metadata["year"])

    # with 50 percent probab join the text with a comma
    if dropout(0.5):
        text = ", ".join(text)
    else:
        text = " ".join(text)

    metadata["text"] = text

    return metadata


def split_audios(metadata_list, split_size=4, split_len=45):

    metadata_new = []

    for metadata in metadata_list:

        # load the audio
        audio = librosa.load(metadata["path"], sr=48000)[0]

        audio_len = audio.shape[0] // 48000

        # check if the audio is long enough
        if audio_len < split_len:
            # split the audio into chunks of 45 seconds into as many chunks as possible
            for i in range(audio_len // split_len):
                audio_new = audio[i * split_len * 48000 : (i + 1) * split_len * 48000]
                metadata_new.append(
                    {
                        "text": metadata["text"] + f" {i+1} of {split_size}",
                        "file_name": "data" + "/" + song_prompt_to_name(metadata["text"] + f" {i+1} of {split_size}") + ".wav",
                    }
                )

                # save the audio
                save_wav(
                    audio_new,
                    audio_folder + "/" + metadata_new[-1]["file_name"],
                    sr=48000,
                )

            # delete the original audio
            os.remove(metadata["path"])

        else:
            # select 4 random 45 second chunks from the audio
            for i in range(split_size):
                start = random.randint(0, audio_len - split_len) * 48000
                audio_new = audio[start : start + split_len * 48000]
                metadata_new.append(
                    {
                        "text": metadata["text"] + f" {i+1} of {split_size}",
                        "file_name": "data" + "/" + song_prompt_to_name(metadata["text"] + f" {i+1} of {split_size}") + ".wav",

                    }
                )

                # save the audio
                save_wav(
                    audio_new,
                    audio_folder + "/" + metadata_new[-1]["file_name"],
                    sr=48000,
                )

            # delete the original audio
            os.remove(metadata["path"])

    return metadata_new


if __name__ == "__main__":

    # create an argument parser
    parser = argparse.ArgumentParser(description="Prepare the dataset for training")

    # add arguments to the parser
    parser.add_argument(
        "--audio_folder",
        type=str,
        default="dataset",
        help="Path to the folder containing the audio files",
    )

    args = parser.parse_args()

    audio_folder = args.audio_folder

    # get the list of audio files
    audio_files = os.listdir(audio_folder)

    # get the metadata for each audio file
    metadata_list = []

    for audio_file in audio_files:
        metadata_current = get_metadata(audio_file)

        if metadata_current is not None:
            metadata_list.append(metadata_current)

    # get the text for each audio file
    metadata_list = [get_text(metadata) for metadata in metadata_list]

    if not os.path.exists(audio_folder + "/data"):
        os.makedirs(audio_folder + "/data")

    # split the audios into 45 second chunks
    metadata_list = split_audios(metadata_list)

    # save the metadata
    save_jsonl(metadata_list, f"{audio_folder}/metadata.jsonl")
    
