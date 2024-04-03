from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2


cfg = get_cfg()
cfg.merge_from_file("./Base-RCNN-VGG-test.yaml")
cfg.MODEL.WEIGHTS = "./output_first_vgg-16/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

predictor = DefaultPredictor(cfg)


img = cv2.imread("./frankfurt_000001_009854_leftImg8bit.jpg")

# 執行推理
outputs = predictor(img)


v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite("eval.jpg", out.get_image()[:, :, ::-1])
