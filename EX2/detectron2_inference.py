from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os

def process_images(input_path, output_path):

    cfg = get_cfg()
    cfg.merge_from_file("/trainingData/sage/DLCV_hentci/EX2/config-files/Best-ResNet50.yaml")
    cfg.MODEL.WEIGHTS = "/trainingData/sage/DLCV_hentci/EX1/output_seventh_r-50/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    predictor = DefaultPredictor(cfg)

    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    for img_file in os.listdir(input_path):
        img_path = os.path.join(input_path, img_file)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')): 
            img = cv2.imread(img_path)
            outputs = predictor(img)

            
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            
            output_img_path = os.path.join(output_path, img_file)
            cv2.imwrite(output_img_path, out.get_image()[:, :, ::-1])


input_path = '/trainingData/sage/DLCV_hentci/img2imgs'
output_path = '/trainingData/sage/DLCV_hentci/annotation_fake_imgs'

process_images(input_path, output_path)


