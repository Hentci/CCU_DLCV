from detectron2.data.datasets import register_pascal_voc
from detectron2.data import DatasetCatalog, MetadataCatalog
# import torch



# print(torch.cuda.is_avaliable())

def register_cityscapes_voc(name, dirname, split, year, class_names):
    register_pascal_voc(
        name, dirname, split, year,
        class_names=class_names
    )

dataset_root = '/trainingData/sage/Cityscapes_dataset/VOC2007/'

class_names = ('truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus')

# register
register_cityscapes_voc("cityscapes_train_voc", dataset_root, "trainval", "2012", class_names)
register_cityscapes_voc("cityscapes_test_voc", dataset_root, "test", "2012", class_names)


def print_dicts():
    dataset_dicts = DatasetCatalog.get("cityscapes_train_voc")
    for d in dataset_dicts:
        print(d)

def print_labels():
    metadata_dicts = MetadataCatalog.get("cityscapes_train_voc")
    print(metadata_dicts.class_names)