import os
import re
import timeit

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from matplotlib.image import imread
import scipy.misc
from PIL import Image  

dataset = "/home/group08/work/coco/train2017/"
#images = os.listdir(dataset)
output_path ="/home/group08/work/Week5/task_d/"
pattern = r'/(.*).yaml'
models = [
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
]

images = ["000000000089", "000000000094", "000000000136", "000000000194", "000000000201","000000000641","000000000643","000000000650","000000000656","000000296170","000000296176","000000296182","000000296187","000000296188","000000296191","000000474862","000000474868","000000474869","000000474256","000000474246","000000474272","000000474279","000000474333","000000474342","000000474353","000000474545","000000509582","000000509588","000000510207","000000511537","000000542154","000000542165","000000543661","000000581929","000000581827","000000341061"]

def createNoisyImage(img,imAux,imName,detection_type, var):
    # Create noisy image
    imA = img[...,::-1]/255.0
    noise = np.random.normal(loc=0, scale =1, size = img.shape)
    noisy = np.clip((imA+noise*(var/10)),0,1)
    noisy = noisy*255
    noisy[imAux==1] = img[imAux==1]
    # Predict with noise
    outputs = predictor(noisy)
    v = Visualizer(noisy[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(imName +"/"+detection_type+"_noise_dot%d_.jpg" % (var), out.get_image()[:, :, ::-1])
    print("noisy image created with %d distortion", var)
    
def overlapColor(img,imAux,imName,color,color_vector):
    imAux2 = np.ones_like(img)
    imAux2[:,:,0] = imAux2[:,:,0]*color_vector[0]
    imAux2[:,:,1] = imAux2[:,:,1]*color_vector[1]
    imAux2[:,:,2] = imAux2[:,:,2]*color_vector[2]
    imAux2[imAux==1] = img[imAux==1]
    # Predict with noise
    imAux2= imAux2.astype(np.uint8)
    outBbox = predictor(imAux2)
    v = Visualizer(imAux2[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outBbox["instances"].to("cpu"))
    cv2.imwrite(str(color)+"/"+str(imName)+".jpg" , out.get_image()[:, :, ::-1])
    
for imName in images: 
    dirName = output_path+str(imName)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    for model in models:
    
        model_name = re.search(pattern, model).group(1)
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
    
        img = cv2.imread(dataset+imName+".jpg")
    
    
        outputs = predictor(img)
        # Visualizer
        
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    
        if model == "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" :
            mask=outputs["instances"].get('pred_masks').to("cpu")[0]
            imAux = mask.numpy().astype(np.uint8)
            imAux2 = np.zeros_like(img)
            imAux2[imAux==1] = img[imAux==1]
            # Predict with noise
            imAux2= imAux2.astype(np.uint8)
            outBbox = predictor(imAux2)
            v = Visualizer(imAux2[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outBbox["instances"].to("cpu"))
            cv2.imwrite("Mask/"+str(imName)+".jpg" , out.get_image()[:, :, ::-1])
            overlapColor(img,imAux, imName,"MaskOrange", [50,120,200])
            overlapColor(img,imAux, imName,"MaskBlue", [200,120,50])
            #cv2.imwrite("maskBlack.jpg",imAux2.astype(np.uint8))
            for i in range(0,11,2):
              createNoisyImage(img,imAux, imName,"Mask", i)
        else:
            mask=outputs["instances"].get('pred_boxes').to("cpu")[0]
            print(mask)
            imAux = cv2.rectangle(np.zeros_like(img), pt1=(mask.tensor.numpy()[0][0],mask.tensor.numpy()[0][1]), pt2=(mask.tensor.numpy()[0][2],mask.tensor.numpy()[0][3]), color=(1,1,1), thickness = -1)
            imAux2 = np.zeros_like(img)
            imAux2[imAux==1] = img[imAux==1]
            # Predict with noise
            imAux2= imAux2.astype(np.uint8)
            outBbox = predictor(imAux2)
            v = Visualizer(imAux2[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outBbox["instances"].to("cpu"))
            cv2.imwrite("BBox/"+str(imName)+".jpg" , out.get_image()[:, :, ::-1])
            overlapColor(img,imAux, imName,"BBoxOrange", [50,120,200])
            overlapColor(img,imAux, imName,"BBoxBlue", [200,120,50])
            #cv2.imwrite("boxBlack.jpg",imAux2.astype(np.uint8))
            for i in range(0,11,2):
              createNoisyImage(img,imAux, imName,"BBox", i)
            
        
    
            
        print("Model completed")
    
print("Sequence complete")
    