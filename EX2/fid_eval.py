from pytorch_fid.fid_score import calculate_fid_given_paths
import os

def calculate_fid(input_path, output_path):
    
    if not os.path.exists(input_path):
        raise ValueError("Input path does not exist or is empty.")
    if not os.path.exists(output_path):
        raise ValueError("Output path does not exist or is empty.")
    

    fid_value = calculate_fid_given_paths([input_path, output_path],
                                          batch_size=1,
                                          device='cuda',  
                                          dims=2048)
    print(f"FID score between input and output images: {fid_value}")


input_folder = '/trainingData/sage/DLCV_hentci/EX2/cityscapes_imgs'
output_folder = '/trainingData/sage/DLCV_hentci/img2imgs'


calculate_fid(input_folder, output_folder)

