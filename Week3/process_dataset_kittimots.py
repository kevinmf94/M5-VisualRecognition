import os
import pickle

from detectron2.structures import BoxMode
from pycocotools import mask
import pandas as pd

KITTIMOTS_DATASET = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_TRAIN = KITTIMOTS_DATASET + "training/image_02/"
KITTIMOTS_TRAIN_INSTANCES = KITTIMOTS_DATASET + "instances_txt/"
TRAIN = ["0000.txt", "0001.txt", "0002.txt", "0003.txt", "0004.txt",
         "0005.txt", "0006.txt", "0007.txt", "0008.txt", "0009.txt"]
VALIDATION = ["0010.txt", "0011.txt"]
TEST = ["0012.txt", "0013.txt", "0014.txt", "0015.txt", "0016.txt", "0017.txt", "0018.txt", "0019.txt", "0020.txt"]


def rle_to_bbox(row):
    rle = {'counts': row.RLE.encode('utf-8'), 'size': [row.Height, row.Width]}
    row.BBox = mask.toBbox(rle)
    return row


def annotations_to_dict(annotations):
    objs = []
    for index, annotation in annotations.iterrows():
        obj = {
            "bbox": annotation.BBox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": annotation.Class - 1
        }

        objs.append(obj)

    return objs


def frame_to_record(sequence, frame, annotations):
    return {
        "file_name": os.path.join(KITTIMOTS_TRAIN, "images/", sequence) + ("/%06d.jpg" % frame),
        "image_id": "%s_%06d" % (sequence, frame),
        "height": annotations.Height.values[0],
        "width": annotations.Width.values[0],
        "annotations": annotations_to_dict(annotations)
    }


def generate_dict(files):
    dataset_dict = []
    for file in files:
        data = pd.read_csv(KITTIMOTS_TRAIN_INSTANCES + file, delimiter=" ",
                           names=["Frame", "Id", "Class", "Height", "Width", "RLE", "BBox"])
        data = data.apply(lambda row: rle_to_bbox(row), axis=1)
        data = data[(data.Class == 1) | (data.Class == 2)]
        sequence = file.split(".")[0]

        print("Processing sequence %s" % sequence, flush=True)

        for frame in data['Frame'].unique():
            record = frame_to_record(sequence, frame, data[data.Frame == frame])
            dataset_dict.append(record)

    return dataset_dict


if __name__ == '__main__':

    print("Generating KITTI-MOTS", flush=True)

    train_records = generate_dict(TRAIN)
    print("Train %d" % len(train_records))
    with open('kittimots_train.dat', 'wb') as f:
        pickle.dump(train_records, f)
        print("Generated kittimots_train.dat", flush=True)

    val_records = generate_dict(VALIDATION)
    print("Val %d" % len(val_records))
    with open('kittimots_val.dat', 'wb') as f:
        pickle.dump(val_records, f)
        print("Generated kittimots_val.dat", flush=True)

    test_records = generate_dict(TEST)
    print("Test %d" % len(test_records))
    with open('kittimots_test.dat', 'wb') as f:
        pickle.dump(test_records, f)
        print("Generated kittimots_test.dat", flush=True)

    print("Generated KITTI-MOTS", flush=True)
