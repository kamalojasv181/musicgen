import shutil
import os 
import glob
import argparse


if __name__ == "__main__":

    # create an argument parser

    parser = argparse.ArgumentParser(description="Prepare the dataset for training")

    # add arguments to the parser

    parser.add_argument("--data_folder", type=str, help="Path to the folder containing the audio files")
    parser.add_argument("--output_folder", type=str, help="Path to the folder where to copy half the files")

    args = parser.parse_args()

    # if the output folder does not exist, create it

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    data_folder = args.data_folder
    output_folder = args.output_folder

    audio_files = glob.glob(os.path.join(data_folder, "*.mp3"))

    # get half the files
    audio_files = audio_files[0:int(len(audio_files)/2)]

    # move the files to the output folder
    for audio_file in audio_files:

        shutil.move(audio_file, output_folder)

    print("Done!")