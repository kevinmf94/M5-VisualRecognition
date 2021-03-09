import timeit

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer

DATASET_TEST = '/home/mcv/datasets/MIT_split/test'
IMG = '/highway/bost154.jpg'

faster_rcnn_models = [
    'faster_rcnn_X_101_32x8d_FPN_3x.yaml',
    'retinanet_R_50_FPN_1x.yaml',
]

thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

# Load IMG
img = cv2.imread(DATASET_TEST + IMG, cv2.IMREAD_COLOR)
cv2.imwrite("input.jpg", img)

for model in faster_rcnn_models:

    for thr in thresholds:
        MODEL = 'COCO-Detection/'+model
        EXPERIMENT_NAME = 'Thr_%s_%s' % (model, thr)

        print(EXPERIMENT_NAME, flush=True)

        # Configuration and prediction
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(MODEL))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr  # set threshold for this model
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thr
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)

        predictor = DefaultPredictor(cfg)

        start = timeit.default_timer()
        outputs = predictor(img)
        stop = timeit.default_timer()
        print('Time inference: ', stop - start)

        #Predicted
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)

        #Visualizer
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("output_%s.jpg" % EXPERIMENT_NAME, out.get_image()[:, :, ::-1])


