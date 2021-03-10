import os
import pickle

import cv2
from detectron2.structures import BoxMode
from sklearn.preprocessing import LabelEncoder

KITTY_DATASET = '/home/mcv/datasets/KITTI/'
TRAIN_IMAGES_LIST = 'train_kitti.txt'
TRAIN_IMG_PATH = os.path.join(KITTY_DATASET, 'data_object_image_2/training/image_2/')
TRAIN_LABELS_PATH = os.path.join(KITTY_DATASET, 'training/label_2/')
CLASS_FIELD = 0
LEFT = 4
TOP = 5
RIGHT = 6
BOTTOM = 7
CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
           'Cyclist', 'Tram', 'Misc', 'DontCare']

label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)


def read_array_file(filename):
    with open(filename) as f:
        data = f.readlines()
        data = [i.replace('\n', '') for i in data]

    return data


def get_kitty_dicts(img_dir):
    train_files = os.path.join(img_dir, TRAIN_IMAGES_LIST)

    labels_files = read_array_file(train_files)

    dataset_dicts = []

    i = 0
    total = len(labels_files)
    for label_file in labels_files:
        annotations = read_array_file(os.path.join(TRAIN_LABELS_PATH, label_file))

        idx = label_file.split(".")[0]
        i += 1
        print("Processing image %d of %d with id %s " % (i, total, idx), flush=True)
        record = {}

        filename = os.path.join(TRAIN_IMG_PATH, idx + '.png')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for annotation in annotations:
            annot_data = annotation.split(" ")

            obj = {
                "bbox": [
                    int(float(annot_data[LEFT])),
                    int(float(annot_data[TOP])),
                    int(float(annot_data[RIGHT])),
                    int(float(annot_data[BOTTOM]))
                ],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": label_encoder.transform([annot_data[CLASS_FIELD]])[0],
            }

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == '__main__':
    print("Starting preprocessing", flush=True)
    train_dict = get_kitty_dicts(KITTY_DATASET)
    print("End preprocessing", flush=True)
    print("Saving dat", flush=True)
    train_dict_file = open('train_dict.dat', 'wb')
    pickle.dump(train_dict, train_dict_file)
    print("dat saved!", flush=True)
