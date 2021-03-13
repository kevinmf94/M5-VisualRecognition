import os

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

KITTY_DATASET = '/home/mcv/datasets/KITTI/'
TEST_IMAGES_LIST = KITTY_DATASET + 'test_kitti.txt'
TEST_IMG_PATH = os.path.join(KITTY_DATASET, 'data_object_image_2/testing/image_2/')
CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
           'Cyclist', 'Tram', 'Misc', 'DontCare']


def read_array_file(filename):
    with open(filename) as f:
        data = f.readlines()
        data = [i.replace('\n', '') for i in data]

    return data


def instance_to_kitty(instance, position):
    # Kitti Output example
    # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01"
    """----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better."""

    bbox = instance.pred_boxes[position].tensor.cpu().data[0]
    score = instance.scores[position].cpu().data
    class_id = instance.pred_classes[position].cpu().data
    class_str = CLASSES[class_id]

    return "%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" \
           % (class_str, 0, 0, 0, bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0, 0, 0, score)


def generate_testing_kitty_files():
    # Load model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    # Generate Test
    test_filenames = read_array_file(TEST_IMAGES_LIST)
    i = 0
    total = len(test_filenames)
    for test_filename in test_filenames:

        idx = test_filename.split(".")[0]
        img_filename = TEST_IMG_PATH + idx + ".png"
        i += 1
        print("Processing image %d of %d with id %s " % (i, total, idx), flush=True)

        print(img_filename)
        img = cv2.imread(img_filename)
        print(img.shape)
        output = predictor(img)

        print(output['instances'])
        print("     ")

        for j in range(len(output['instances'])):
            kitti_str = instance_to_kitty(output['instances'], j)
            print(kitti_str)
            exit()


if __name__ == '__main__':
    generate_testing_kitty_files()
