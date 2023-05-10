from frechet_audio_distance import FrechetAudioDistance
import argparse
import sys

sys.path.append("../")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=str, required=True, help="Path to the background audio directory")
    parser.add_argument("--eval", type=str, required=True, help="Path to the eval audio directory")


    # to use `PANN`

    frechet = FrechetAudioDistance(
        model_name="pann",
        use_pca=False, 
        use_activation=False,
        verbose=False
    )

    fad_score = frechet.score(
        background_dir=args.background,
        eval_dir=args.eval
    )

    print(fad_score)