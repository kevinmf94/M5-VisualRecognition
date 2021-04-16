import os
import re
import timeit

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer

dataset = "/home/group08/mcv/datasets/out_of_context"
images = os.listdir(dataset)
pattern = r'/(.*).yaml'
models = [
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
]

for model in models:

    model_name = re.search(pattern, model).group(1)
    os.mkdir(model_name)
    print(model_name, flush=True)

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

    for image in images:

        img = cv2.imread("%s/%s" % (dataset, image))

        start = timeit.default_timer()
        outputs = predictor(img)
        stop = timeit.default_timer()
        print('Time inference: ', stop - start)

        # Visualizer
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("%s/output_%s.jpg" % (model_name, image.replace(".jpg", "")), out.get_image()[:, :, ::-1])