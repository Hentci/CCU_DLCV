from PIL import Image
import os

source_dir = '/trainingData/sage/Cityscapes_dataset/VOC2007/JPEGImages_png/'
target_dir = '/trainingData/sage/Cityscapes_dataset/VOC2007/JPEGImages/'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for filename in os.listdir(source_dir):
    if filename.endswith('.png'):
        with Image.open(os.path.join(source_dir, filename)) as img:
            target_filename = filename[:-4] + '.jpg'
            img.save(os.path.join(target_dir, target_filename), "JPEG")
