import pickle

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo

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


def load_test_dataset():
    with open('../../kittimots_test_coco.dat', 'rb') as f:
        val_data = pickle.load(f)
    return val_data


DatasetCatalog.register("kittimots_test_coco", load_test_dataset)
coco_metadata = MetadataCatalog.get("coco_2017_val")
MetadataCatalog.get("kittimots_test_coco").set(thing_classes=coco_metadata.thing_classes)

for config in configs.keys():
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

        trainer = DefaultPredictor(cfg)

        # Evaluate
        val_loader = build_detection_test_loader(cfg, "kittimots_test_coco")
        evaluator = COCOEvaluator("kittimots_test_coco", ("bbox", "segm"), False, output_dir="./output_%s/" % model)
        print(inference_on_dataset(trainer.model, val_loader, evaluator))


