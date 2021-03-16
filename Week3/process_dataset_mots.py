import os
import pickle

from detectron2.structures import BoxMode
from pycocotools import mask
import pandas as pd

MOTS_DATASET = '/home/mcv/datasets/MOTSChallenge/'
MOTS_TRAIN = MOTS_DATASET + "train/"
MOTS_TRAIN_INSTANCES = MOTS_DATASET + "train/instances_txt/"
TRAIN = ["0002.txt", "0005.txt", "0009.txt"]
VALIDATION = ["0011.txt"]


def rle_to_bbox(row):
    rle = {'counts': row.RLE.encode('utf-8'), 'size': [row.Height, row.Width]}
    row.BBox = mask.toBbox(rle)
    row.Class = 0
    return row


def annotations_to_dict(annotations):

    objs = []
    for index, annotation in annotations.iterrows():

        obj = {
            "bbox": annotation.BBox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": annotation.Class
        }

        objs.append(obj)

    return objs


def frame_to_record(sequence, frame, annotations):
    return {
        "file_name": os.path.join(MOTS_TRAIN, "images/", sequence) + ("/%06d.jpg" % frame),
        "image_id": "%s_%06d" % (sequence, frame),
        "height": annotations.Height.values[0],
        "width": annotations.Width.values[0],
        "annotations": annotations_to_dict(annotations)
    }


def generate_dict(files):
    dataset_dict = []
    for file in files:
        data = pd.read_csv(MOTS_TRAIN_INSTANCES + file, delimiter=" ",
                           names=["Frame", "Id", "Class", "Height", "Width", "RLE", "BBox"])
        data = data.apply(lambda row: rle_to_bbox(row), axis=1)
        sequence = file.split(".")[0]

        print("Processing sequence %s" % sequence, flush=True)

        for frame in data['Frame'].unique():
            record = frame_to_record(sequence, frame, data[data.Frame == frame])
            dataset_dict.append(record)

    return dataset_dict


if __name__ == '__main__':

    print("Generating MOTSChallenge", flush=True)

    train_records = generate_dict(TRAIN)
    print("Train %d" % len(train_records))
    with open('mots_train.dat', 'wb') as f:
        pickle.dump(train_records, f)
        print("Generated mots_train.dat", flush=True)

    val_records = generate_dict(VALIDATION)
    print("Val %d" % len(val_records))
    with open('mots_validation.dat', 'wb') as f:
        pickle.dump(val_records, f)
        print("Generated mots_validation.dat", flush=True)

    print("Generated MOTSChallenge", flush=True)
