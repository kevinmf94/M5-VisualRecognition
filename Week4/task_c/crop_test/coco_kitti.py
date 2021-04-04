import os
import pickle

from detectron2.evaluation import COCOEvaluator

import torch
from detectron2.config import get_cfg, CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.model_zoo import model_zoo
from detectron2.utils import comm

MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"

TRAIN_DAT = '../kittimots_train.dat'
VAL_DAT = '../kittimots_val.dat'
TRAIN_NAME = 'kittimots_train'
VAL_NAME = 'kittimots_val'
CLASSES = ['Cars', 'Pedestrian']

CROP_TH = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9]


# Dataset
def load_train_dataset():
    train_dict_file = open(TRAIN_DAT, 'rb')
    train_dict = pickle.load(train_dict_file)
    return train_dict


def load_val_dataset():
    val_dict_file = open(VAL_DAT, 'rb')
    val_dict = pickle.load(val_dict_file)
    return val_dict


DatasetCatalog.register(TRAIN_NAME, load_train_dataset)
DatasetCatalog.register(VAL_NAME, load_val_dataset)
MetadataCatalog.get(TRAIN_NAME).set(thing_classes=CLASSES)
MetadataCatalog.get(VAL_NAME).set(thing_classes=CLASSES)


# ValidationLoss
class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)


for cropThX in CROP_TH:
    for cropThY in CROP_TH:
        print("Test [%f, %f]" % (cropThX, cropThY))

        # Model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(MODEL))
        cfg.DATASETS.TRAIN = (TRAIN_NAME,)
        cfg.DATASETS.TEST = (VAL_NAME,)
        cfg.TEST.EVAL_PERIOD = 100
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
        cfg.SOLVER.MAX_ITER = 200  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
        cfg.INPUT.MASK_FORMAT = 'bitmask'

        cfg.INPUT.CROP = CfgNode({"ENABLED": False})
        cfg.INPUT.CROP.TYPE = "relative_range"
        cfg.INPUT.CROP.SIZE = [cropThX, cropThY]

        # Train
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        val_loss = ValidationLoss(cfg)
        trainer.register_hooks([val_loss])
        # swap the order of PeriodicWriter and ValidationLossl
        trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
        trainer.resume_or_load(resume=False)
        trainer.train()

        # load weights
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

        # Set training data-set path
        cfg.DATASETS.TEST = (VAL_NAME,)

        evaluator = COCOEvaluator(VAL_NAME, cfg, False, output_dir="./output_%f_%f/" % (cropThX, cropThY))
        trainer.test(cfg, trainer.model, evaluators=[evaluator])
        print("evaluated")
