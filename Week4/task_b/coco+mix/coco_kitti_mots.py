import os
import pickle
import timeit

from detectron2.evaluation import COCOEvaluator 
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer
import cv2
from numpy import random

TRAIN_DAT = '../../mots_kitti_train.dat'
VAL_DAT = '../../kittimots_test.dat'
TRAIN_NAME = 'mots_kitti_train'
VAL_NAME = 'kittimots_test'
CLASSES = ['Cars','Pedestrian']


def load_test_dataset():
    with open('../kittimots_test.dat', 'rb') as f:
        val_data = pickle.load(f)
    return val_data

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

# Model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = (TRAIN_NAME,)
cfg.DATASETS.TEST = (VAL_NAME,)
cfg.TEST.EVAL_PERIOD = 100
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
cfg.INPUT.MASK_FORMAT = 'bitmask'


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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

# Set training data-set path
cfg.DATASETS.TEST = (VAL_NAME,)
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator(VAL_NAME, cfg, False, output_dir="./output/")
trainer.test(cfg, trainer.model, evaluators=[evaluator])
print("evaluated")

       
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

