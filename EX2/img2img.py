from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline

# 還不錯參數
'''
python3 train_dreambooth.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --instance_data_dir /trainingData/sage/DLCV_hentci/EX2/cityscapes_imgs --output_dir /trainingData/sage/DLCV_hentci/fine_tuned_model --instance_prompt "Image similar to Cityscapes" --num_train_epochs 15 --learning_rate 2e-6 --train_batch_size 2 
'''

'''
DreamBooth 主要設置 (50epoch 可能都不夠...)
python3 train_dreambooth.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --instance_data_dir /trainingData/sage/DLCV_hentci/EX2/cityscapes_imgs --output_dir /trainingData/sage/DLCV_hentci/fine_tuned_model --instance_prompt "Image similar to Cityscapes" --num_train_epochs 300 --learning_rate 2e-6 --train_batch_size 1 
'''

# FID
'''
CUDA_VISIBLE_DEVICE=1 python -m pytorch_fid /trainingData/sage/DLCV_hentci/EX2/cityscapes_imgs /trainingData/sage/DLCV_hentci/img2imgs
'''

# 指定模型的路径
model_path = "/trainingData/sage/DLCV_hentci/fine_tuned_model"

# 加载模型
dreambooth_generator = StableDiffusionPipeline.from_pretrained(model_path)
dreambooth_generator.safety_checker = None  


prompt = "Image similar to Cityscapes"


for i in range(1000):
    # Generate the image with the prompt
    output = dreambooth_generator(prompt)
    # Get the image from the output
    image = output["images"][0]  # Assuming the output is a dict with an 'images' key
    # Save the image to a file
    image.save(f"/trainingData/sage/DLCV_hentci/img2imgs/generated_image_{i}.png")


