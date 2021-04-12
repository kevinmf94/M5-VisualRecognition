import os
import pickle
import timeit
import cv2

from detectron2.evaluation import COCOEvaluator 

import torch
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.model_zoo import model_zoo
from detectron2.utils import comm
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer



TRAIN_DAT = '../../kittimots_train.dat'
VAL_DAT = '../../kittimots_test_coco.dat'
TRAIN_NAME = 'kittimots_train'
VAL_NAME = 'kittimots_test'
CLASSES = ['Cars','Pedestrian']

# Dataset
def load_train_dataset():
    train_dict_file = open(TRAIN_DAT, 'rb')
    train_dict = pickle.load(train_dict_file)
    return train_dict


def load_val_dataset():
    val_dict_file = open(VAL_DAT, 'rb')
    val_dict = pickle.load(val_dict_file)
    return val_dict

DatasetCatalog.register(VAL_NAME, load_val_dataset)
coco_metadata = MetadataCatalog.get("coco_2017_val")
MetadataCatalog.get(VAL_NAME).set(thing_classes=coco_metadata.thing_classes)

# Model

# Configuration and prediction
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
cfg.DATALOADER.NUM_WORKERS = 4

trainer = DefaultPredictor(cfg)
predictor = DefaultPredictor(cfg)

# Evaluate
val_loader = build_detection_test_loader(cfg, VAL_NAME)
evaluator = COCOEvaluator(VAL_NAME, ("bbox", "segm"), False, output_dir="./output_%s/" % "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
print(inference_on_dataset(trainer.model, val_loader, evaluator))


os.makedirs("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x")

Path = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
items = ['0015/000152', '0016/000027', '0019/000115', '0019/000222', '0019/000324', '0019/000570',
            '0019/001035', '0020/000630', '0020/000688']

for i in items:

    img = cv2.imread(Path + i + '.png')
    print(i)

    start = timeit.default_timer()
    outputs = predictor(img)
    stop = timeit.default_timer()
    print('Time inference: ', stop - start)

    # Visualizer
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("%s/output_%s.jpg" % ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x", i[5:]), out.get_image()[:, :, ::-1])
