import json
import cv2
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_pascal_voc

def register_cityscapes_voc(name, dirname, split, year, class_names):
    register_pascal_voc(
        name, dirname, split, year,
        class_names=class_names
    )
    MetadataCatalog.get(name).set(evaluator_type="coco")

dataset_root = '/trainingData/sage/Cityscapes_dataset/VOC2007/'

class_names = ('truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus')


# register
def register_cityscapes():
    register_cityscapes_voc("cityscapes_train_voc", dataset_root, "trainval", "2012", class_names)
    register_cityscapes_voc("cityscapes_test_voc", dataset_root, "test", "2012", class_names)


def create_coco_json(predictor, input_path, output_json_path):
    # 初始化 COCO 数据字典
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 填充类别信息（根据实际情况调整）
    for idx, category_name in enumerate(predictor.metadata.thing_classes):
        coco_output["categories"].append({
            "id": idx + 1,
            "name": category_name
        })

    annotation_id = 1  # 注释的唯一ID
    for img_id, img_file in enumerate(os.listdir(input_path), start=1):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_path, img_file)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")

            # 图像信息
            coco_output["images"].append({
                "id": img_id,
                "file_name": img_file,
                "height": height,
                "width": width
            })

            # 注释信息
            boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": cls + 1,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1

    # 保存到 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_output, f)

# 配置和模型加载
cfg = get_cfg()
cfg.merge_from_file("/trainingData/sage/DLCV_hentci/EX2/config-files/Best-ResNet50.yaml")
cfg.MODEL.WEIGHTS = "/trainingData/sage/DLCV_hentci/EX1/output_seventh_r-50/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

register_cityscapes()

# 输入输出路径
input_path = '/trainingData/sage/DLCV_hentci/img2imgs'
output_json_path = '/trainingData/sage/DLCV_hentci/COCO_annotation/coco_annotations.json'

create_coco_json(predictor, input_path, output_json_path)