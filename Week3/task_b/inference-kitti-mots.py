import pickle
import timeit

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer

faster_rcnn_models = [
    'faster_rcnn_X_101_32x8d_FPN_3x.yaml',
    'retinanet_R_50_FPN_1x.yaml',
]

with open('../kittimots_test.dat', 'rb') as f:
    val_data = pickle.load(f)

CLASSES = ['Car', 'Pedestrian']
INFERENCE_FRAMES = 60

for item in val_data:

    frame_id = int(item['file_name'].split("/")[-1].split(".")[0])
    if frame_id % INFERENCE_FRAMES != 0:
        continue

    # Load IMG
    print(item['file_name'])
    img = cv2.imread(item['file_name'])

    for model in faster_rcnn_models:

        MODEL = 'COCO-Detection/'+model
        EXPERIMENT_NAME = 'Inference_%s_%s' % (model, item['image_id'])

        print(EXPERIMENT_NAME, flush=True)

        # Configuration and prediction
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(MODEL))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
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


