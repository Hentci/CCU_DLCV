from pytorch_fid.fid_score import calculate_fid_given_paths
import os

def calculate_fid(input_path, output_path):
    # 确保路径中存在图像
    if not os.path.exists(input_path):
        raise ValueError("Input path does not exist or is empty.")
    if not os.path.exists(output_path):
        raise ValueError("Output path does not exist or is empty.")
    
    # 计算 FID 分数
    fid_value = calculate_fid_given_paths([input_path, output_path],
                                          batch_size=1,
                                          device='cuda',  # 使用 'cuda' 或 'cpu'
                                          dims=2048)
    print(f"FID score between input and output images: {fid_value}")

# 输入和输出图像的路径
input_folder = '/trainingData/sage/DLCV_hentci/EX2/cityscapes_imgs'
output_folder = '/trainingData/sage/DLCV_hentci/img2imgs'

# 调用函数计算 FID 分数
calculate_fid(input_folder, output_folder)

