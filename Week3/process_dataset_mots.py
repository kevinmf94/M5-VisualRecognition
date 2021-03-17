import os
import pickle

from detectron2.structures import BoxMode
from pycocotools import mask
import pandas as pd

MOTS_DATASET = '/home/mcv/datasets/MOTSChallenge/'
MOTS_TRAIN_IMG = MOTS_DATASET + "train/images/"
MOTS_TRAIN_INSTANCES = MOTS_DATASET + "train/instances_txt/"
TRAIN = ["0002", "0005", "0009"]
VALIDATION = ["0011"]


def rle_to_bbox(row):
    rle = {'counts': row.RLE.encode('utf-8'), 'size': [row.Height, row.Width]}
    row.BBox = mask.toBbox(rle)
    row.Class = 0
    return row


def annotations_to_dict(annotations):

    objs = []
    for index, annotation in annotations.iterrows():

        obj = {
            "bbox": annotation.BBox.astype(int).tolist(),
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": annotation.Class
        }

        objs.append(obj)

    return objs


def frame_to_record(sequence, frame, annotations, size):
    return {
        "file_name": os.path.join(MOTS_TRAIN_IMG, sequence) + ("/%06d.jpg" % frame),
        "image_id": "%s_%06d" % (sequence, frame),
        "height": size[0],
        "width": size[1],
        "annotations": annotations_to_dict(annotations)
    }


def generate_dict(sequences):
    dataset_dict = []

    for sequence in sequences:
        images = os.listdir(os.path.join(MOTS_TRAIN_IMG, sequence))
        frames = len(images)
        images.sort()

        data = pd.read_csv(MOTS_TRAIN_INSTANCES + sequence + ".txt", delimiter=" ",
                           names=["Frame", "Id", "Class", "Height", "Width", "RLE", "BBox"])
        data = data.apply(lambda row: rle_to_bbox(row), axis=1)
        height = data.Height[0]
        width = data.Width[0]

        print("Processing sequence %s" % sequence, flush=True)
        for frame in range(frames):
            print("Processing frame %s" % frame, flush=True, end="\r")
            annotations = data[data.Frame == frame]
            annotations = annotations[(annotations.Class == 2)]
            record = frame_to_record(sequence, frame, annotations, (height, width))
            dataset_dict.append(record)

    return dataset_dict


if __name__ == '__main__':

    print("Generating MOTSChallenge", flush=True)

    train_records = generate_dict(TRAIN)
    print("\nTrain %d" % len(train_records))
    with open('mots_train.dat', 'wb') as f:
        pickle.dump(train_records, f)
        print("Generated mots_train.dat", flush=True)

    val_records = generate_dict(VALIDATION)
    print("\nVal %d" % len(val_records))
    with open('mots_validation.dat', 'wb') as f:
        pickle.dump(val_records, f)
        print("Generated mots_validation.dat", flush=True)

    print("Generated MOTSChallenge", flush=True)
