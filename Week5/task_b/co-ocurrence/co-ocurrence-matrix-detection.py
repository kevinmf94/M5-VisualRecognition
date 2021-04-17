import os

import cv2
import numpy as np
import pandas as pd
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

dataset = "/home/group08/work/coco/train2017"
images = os.listdir(dataset)
model = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"

coco = MetadataCatalog.get('coco_2017_train')
dataset_classes = coco.thing_classes
n_classes = len(dataset_classes)
print("COCO CLASSES %d" % n_classes)
print(dataset_classes)
print("COCO CLASSES END", end='\n\n')

print(model, flush=True)

# Configuration and prediction
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
cfg.DATALOADER.NUM_WORKERS = 4

predictor = DefaultPredictor(cfg)

comatrix = np.zeros((n_classes, n_classes), dtype=np.uint)
total = len(images)
cont = 0
for image in images:
    cont += 1
    print('Img %d of %d Img: %s' % (cont, total, image), end='\r', flush=True)
    img = cv2.imread("%s/%s" % (dataset, image))
    outputs = predictor(img)

    predicted = outputs['instances'].pred_classes.cpu().numpy()
    classes_pred = len(predicted)
    for i in range(len(predicted)):
        actual = predicted[i]
        mask = np.ones(classes_pred, dtype=np.bool)
        mask[i] = False
        others = predicted[mask]

        for co_ocurrence in others:
            comatrix[actual][co_ocurrence] += 1

print("Generate comatrix", flush=True)
data = pd.DataFrame(comatrix, columns=dataset_classes, index=dataset_classes)
data.to_csv('comatrix_detection.csv')
print("Comatrix generated!", flush=True)