import os
import pickle
import random

import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer

CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
           'Cyclist', 'Tram', 'Misc', 'DontCare']


def load_dataset():
    train_dict_file = open('train_dict.dat', 'rb')
    train_dict = pickle.load(train_dict_file)
    return train_dict


DatasetCatalog.register("kitty_train", load_dataset)
MetadataCatalog.get("kitty_train").set(thing_classes=CLASSES)
kitty_metadata = MetadataCatalog.get("kitty_train")

dataset_dicts = load_dataset()
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=kitty_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite(d["image_id"]+".png", out.get_image()[:, :, ::-1])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("kitty_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

print("Start traininig...", flush=True)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("End traininig!", flush=True)
