import os
import pickle
import sys

from detectron2.structures import BoxMode
from pycocotools import mask
import pandas as pd

KITTIMOTS_DATASET = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_IMAGES = KITTIMOTS_DATASET + "training/image_02/"
KITTIMOTS_IMG_INSTANCES = KITTIMOTS_DATASET + "instances_txt/"
TRAIN = ["0000", "0001", "0002", "0003", "0004",
         "0005", "0006", "0007", "0008", "0009"]
VALIDATION = ["0010", "0011"]
TEST = ["0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]

# In COCO Car = 3, Person = 1 (Pedestrian)
PARSE_CLASS = {1: 3, 2: 1}

def rle_to_bbox(row):
    rle = {'counts': row.RLE.encode('utf-8'), 'size': [row.Height, row.Width]}
    row.BBox = mask.toBbox(rle)
    return row


def annotations_to_dict(annotations):
    objs = []
    for index, annotation in annotations.iterrows():
        obj = {
            "bbox": annotation.BBox.astype(int),
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": PARSE_CLASS[annotation.Class]
        }

        objs.append(obj)

    return objs


def frame_to_record(sequence, frame, annotations, size):
    return {
        "file_name": os.path.join(KITTIMOTS_IMAGES, sequence) + ("/%06d.png" % frame),
        "image_id": "%s_%06d" % (sequence, frame),
        "height": size[0],
        "width": size[1],
        "annotations": annotations_to_dict(annotations)
    }


def generate_dict(sequences):
    dataset_dict = []

    for sequence in sequences:
        images = os.listdir(os.path.join(KITTIMOTS_IMAGES, sequence))
        frames = len(images)
        images.sort()

        data = pd.read_csv(KITTIMOTS_IMG_INSTANCES + sequence + ".txt", delimiter=" ",
                           names=["Frame", "Id", "Class", "Height", "Width", "RLE", "BBox"])
        data = data.apply(lambda row: rle_to_bbox(row), axis=1)
        height = data.Height[0]
        width = data.Width[0]
        print("Processing sequence %s" % sequence, flush=True)

        for frame in range(frames):
            print("Processing frame %s" % frame, flush=True, end="\r")
            annotations = data[data.Frame == frame]
            annotations = annotations[(annotations.Class == 1) | (annotations.Class == 2)]
            record = frame_to_record(sequence, frame, annotations, (height, width))
            dataset_dict.append(record)

    return dataset_dict


if __name__ == '__main__':

    print("Generating KITTI-MOTS", flush=True)

    test_records = generate_dict(TEST)
    print("\nTest %d" % len(test_records))
    with open('kittimots_test_taskc.dat', 'wb') as f:
        pickle.dump(test_records, f)
        print("Generated kittimots_test_taskc.dat", flush=True)

    print("Generated KITTI-MOTS", flush=True)
