import glob
import json

import matplotlib.image as mpimg

from transform_images import strip_transparency


def load_dataset() -> list:
    dataset_directory = '../emoji-data'

    # filter only person emojis without tones
    with open('{dataset_directory}/emoji.json'.format(dataset_directory=dataset_directory)) as data_file:
        emojis = json.load(data_file)

    # Normal version of dataset: 37/148 emojis
    # smileys = [e for e in emojis if e["category"] == "Smileys & Emotion" and "face" in e["short_name"]]

    # Extended version of dataset: 100/400 emojis
    smileys = [e for e in emojis if
               e['category'] == 'Smileys & Emotion' and
               e['name'] is not None and
               'FACE' in e['name']]

    print("minimized dictionary: {} left out of {} emojis".format(len(smileys), len(emojis)))

    # filter all files for the given smiley unicode
    paths = [
        "{dataset_directory}/img-apple-64".format(dataset_directory=dataset_directory),
        "{dataset_directory}/img-facebook-64".format(dataset_directory=dataset_directory),
        "{dataset_directory}/img-google-64".format(dataset_directory=dataset_directory),
        "{dataset_directory}/img-twitter-64".format(dataset_directory=dataset_directory)
    ]
    file_names = [s["image"] for s in smileys]
    file_paths = []
    for p in paths:
        image_paths = glob.glob(p + "/*.png")
        filtered = [p for p in image_paths if p.split("\\")[-1] in file_names]
        file_paths = file_paths + filtered

    # load all files
    images = [mpimg.imread(f) for f in file_paths]
    images = [i for i in images if i.shape == (64, 64, 4)]
    images = [strip_transparency(i) for i in images]

    return images
