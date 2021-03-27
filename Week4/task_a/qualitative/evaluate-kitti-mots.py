import os
import pickle
import timeit

import cv2
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
from numpy import random

configs = {
    "COCO-InstanceSegmentation": [
        "mask_rcnn_R_101_C4_3x.yaml",
        "mask_rcnn_R_101_DC5_3x.yaml",
        "mask_rcnn_R_101_FPN_3x.yaml",
        "mask_rcnn_R_50_C4_1x.yaml",
        "mask_rcnn_R_50_C4_3x.yaml",
        "mask_rcnn_R_50_DC5_1x.yaml",
        "mask_rcnn_R_50_DC5_3x.yaml",
        "mask_rcnn_R_50_FPN_1x.yaml",
        "mask_rcnn_R_50_FPN_3x.yaml",
        "mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    ],
    "Cityscapes": [
        "mask_rcnn_R_50_FPN.yaml"
    ]
}

with open('../../kittimots_test.dat', 'rb') as f:
    val_data = pickle.load(f)

rand_items = random.randint(len(val_data), size=(10))

for config in configs.keys():

    os.mkdir(config)
    for model in configs[config]:

        MODEL = '%s/%s' % (config, model)

        print(MODEL, flush=True)

        # Configuration and prediction
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(MODEL))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
        cfg.DATALOADER.NUM_WORKERS = 4

        predictor = DefaultPredictor(cfg)

        os.mkdir(config+"/"+model)
        for i in rand_items:

            img = cv2.imread(val_data[i]['file_name'])
            print(val_data[i]['image_id'])

            start = timeit.default_timer()
            outputs = predictor(img)
            stop = timeit.default_timer()
            print('Time inference: ', stop - start)

            # Visualizer
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite("%s/output_%s.jpg" % (MODEL, val_data[i]['image_id']), out.get_image()[:, :, ::-1])


