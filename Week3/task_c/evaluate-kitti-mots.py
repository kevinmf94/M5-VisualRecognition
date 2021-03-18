import pickle

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog, DatasetMapper
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo

faster_rcnn_models = [
    'faster_rcnn_X_101_32x8d_FPN_3x.yaml',
    'retinanet_R_50_FPN_1x.yaml',
]


def load_test_dataset():
    with open('../kittimots_test_taskc.dat', 'rb') as f:
        val_data = pickle.load(f)
    return val_data


CLASSES = ['Car', 'Person']

DatasetCatalog.register("kittimots_test", load_test_dataset)
coco_metadata = MetadataCatalog.get("coco_2017_val")
MetadataCatalog.get("kittimots_test").set(thing_classes=coco_metadata.thing_classes)

for model in faster_rcnn_models:

    MODEL = 'COCO-Detection/'+model

    print(MODEL, flush=True)

    # Configuration and prediction
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.DATALOADER.NUM_WORKERS = 4

    trainer = DefaultPredictor(cfg)

    # Evaluate
    evaluator = COCOEvaluator("kittimots_test", cfg, False, output_dir="./output_%s/" % model)
    val_loader = build_detection_test_loader(cfg, "kittimots_test")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    print("Finish %s" % MODEL, flush=True)


