import glob
import os
import pickle

from detectron2.structures import BoxMode
from pycocotools import mask
import pandas as pd

MOTS_DATASET = '/home/mcv/datasets/MOTSChallenge/'
MOTS_TRAIN_IMG = MOTS_DATASET + "train/images/"
MOTS_TRAIN_INSTANCES = MOTS_DATASET + "train/instances_txt/"
TRAIN_MOTS = ["0002", "0005", "0009"]
VALIDATION_MOTS = ["0011"]

KITTIMOTS_DATASET = '/home/mcv/datasets/KITTI-MOTS/'
KITTIMOTS_IMAGES = KITTIMOTS_DATASET + "training/image_02/"
KITTIMOTS_IMG_INSTANCES = KITTIMOTS_DATASET + "instances_txt/"
TRAIN_KITTI = ["0000", "0001", "0002", "0003", "0004",
         "0005", "0006", "0007", "0008", "0009"]
VALIDATION_KITTI = ["0010", "0011"]


def rle_to_bbox_k(row):
    rle = {'counts': row.RLE.encode('utf-8'), 'size': [row.Height, row.Width]}
    row.BBox = mask.toBbox(rle)
    return row


def annotations_to_dict_k(annotations):
    objs = []
    for index, annotation in annotations.iterrows():
        obj = {
            "bbox": annotation.BBox.astype(int).tolist(),
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": annotation.Class - 1
        }

        objs.append(obj)

    return objs

def rle_to_bbox_m(row):
    rle = {'counts': row.RLE.encode('utf-8'), 'size': [row.Height, row.Width]}
    row.BBox = mask.toBbox(rle)
    row.Class = 1
    return row


def annotations_to_dict_m(annotations):
    objs = []
    for index, annotation in annotations.iterrows():
        obj = {
            "bbox": annotation.BBox.astype(int).tolist(),
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": annotation.Class
        }

        objs.append(obj)

    return objs

def frame_to_record(sequence, frame, annotations, size, img_Path):
    return {
        "file_name": os.path.join(img_Path, sequence) + ("/%06d.jpg" % frame),
        "image_id": "%s_%06d" % (sequence, frame),
        "height": size[0],
        "width": size[1],
        "annotations": annotations_to_dict_m(annotations)
    }

def frame_to_record_KITTI(sequence, frame, annotations, size, img_Path):
    return {
        "file_name": os.path.join(img_Path, sequence) + ("/%06d.png" % frame),
        "image_id": "%s_%06d" % (sequence, frame),
        "height": size[0],
        "width": size[1],
        "annotations": annotations_to_dict_k(annotations)
    }


def generate_dict(sequences_MOTS, sequences_KITTI):
    dataset_dict = []

    for sequence in sequences_MOTS:
        images = glob.glob(os.path.join(MOTS_TRAIN_IMG, sequence) + "/*.jpg")
        frames = len(images)
        images.sort()

        data = pd.read_csv(MOTS_TRAIN_INSTANCES + sequence + ".txt", delimiter=" ",
                           names=["Frame", "Id", "Class", "Height", "Width", "RLE", "BBox"])
        data = data[(data.Class == 2)]
        data = data.apply(lambda row: rle_to_bbox_m(row), axis=1)
        height = data.Height[0]
        width = data.Width[0]

        print("Processing sequence %s" % sequence, flush=True)
        for frame in range(1, frames):
            print("Processing frame %s" % frame, flush=True, end="\r")
            annotations = data[data.Frame == frame]
            record = frame_to_record(sequence, frame, annotations, (height, width),MOTS_TRAIN_IMG)
            dataset_dict.append(record)

    for sequence in sequences_KITTI:
        images = os.listdir(os.path.join(KITTIMOTS_IMAGES, sequence))
        frames = len(images)
        images.sort()

        data = pd.read_csv(KITTIMOTS_IMG_INSTANCES + sequence + ".txt", delimiter=" ",
                           names=["Frame", "Id", "Class", "Height", "Width", "RLE", "BBox"])
        data = data.apply(lambda row: rle_to_bbox_k(row), axis=1)
        height = data.Height[0]
        width = data.Width[0]
        print("Processing sequence %s" % sequence, flush=True)

        for frame in range(frames):
            print("Processing frame %s" % frame, flush=True, end="\r")
            annotations = data[data.Frame == frame]
            annotations = annotations[(annotations.Class == 1) | (annotations.Class == 2)]
            record = frame_to_record_KITTI(sequence, frame, annotations, (height, width), KITTIMOTS_IMAGES)
            dataset_dict.append(record)

    return dataset_dict


if __name__ == '__main__':

    print("Generating MOT KITTI", flush=True)

    train_records = generate_dict(TRAIN_MOTS, TRAIN_KITTI)
    print("\nTrain %d" % len(train_records))
    with open('mots_kitti_train.dat', 'wb') as f:
        pickle.dump(train_records, f)
        print("Generated mots_kitti_train.dat", flush=True)

    val_records = generate_dict(VALIDATION_MOTS, VALIDATION_KITTI)
    print("\nVal %d" % len(val_records))
    with open('mots_kitti_validation.dat', 'wb') as f:
        pickle.dump(val_records, f)
        print("Generated mots_kitti_validation.dat", flush=True)

    print("Generated MOTS KITTI Challenge", flush=True)
