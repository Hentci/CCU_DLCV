from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2


cfg = get_cfg()
cfg.merge_from_file("/trainingData/sage/DLCV_hentci/EX2/config-files/Best-ResNet50.yaml")
cfg.MODEL.WEIGHTS = "/trainingData/sage/DLCV_hentci/EX1/output_seventh_r-50/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

predictor = DefaultPredictor(cfg)


img = cv2.imread("/trainingData/sage/DLCV_hentci/img2imgs/generated_image_0.png")

# 執行推理
outputs = predictor(img)


v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite("eval.jpg", out.get_image()[:, :, ::-1])
