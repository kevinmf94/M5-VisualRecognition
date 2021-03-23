import pickle
import os

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog, DatasetMapper
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo

from detectron2.utils.visualizer import ColorMode


def load_test_dataset():
    with open('../kittimots_test.dat', 'rb') as f:
        val_data = pickle.load(f)
    return val_data

CLASSES = ['Cars', 'Pedestrian']


DatasetCatalog.register("kittimots_test", load_test_dataset)
MetadataCatalog.get("kittimots_test").set(thing_classes=CLASSES)

MODEL = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
print(MODEL, flush=True)

# Configuration and prediction
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(MODEL))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# load weights
cfg.MODEL.WEIGHTS = os.path.join("output/model_final.pth")
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)

trainer = DefaultPredictor(cfg)

# Set training data-set path
cfg.DATASETS.TEST = ("kittimots_test",)

# Evaluate
evaluator = COCOEvaluator("kittimots_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "kittimots_test")
print(inference_on_dataset(trainer.model, val_loader, [evaluator]))


