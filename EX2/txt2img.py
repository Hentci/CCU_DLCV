from diffusers import DiffusionPipeline
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

def load_models():
    # Load diffusion models
    model_1 = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir="/trainingData/sage/DLCV_hentci/model_cache/")
    model_2 = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", cache_dir="/trainingData/sage/DLCV_hentci/model_cache/")
    model_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", cache_dir="/trainingData/sage/DLCV_hentci/model_cache/")

    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/trainingData/sage/DLCV_hentci/model_cache/")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/trainingData/sage/DLCV_hentci/model_cache/")

    return model_1, model_2, model_3, clip_model, clip_processor

def generate_image(model, prompt, save_path, image_index):
    if model is not None:
        with torch.no_grad():
            result = model(prompt=prompt)
            if 'images' in result:
                image = result['images'][0]  # Assuming 'images' is the correct key and it contains a list of images
                
                # Check if image needs conversion
                if isinstance(image, torch.Tensor):
                    # Convert from tensor to PIL Image
                    image = Image.fromarray(image.permute(1, 2, 0).numpy().astype("uint8"))
                elif not isinstance(image, Image.Image):
                    raise TypeError("The output image is neither a PIL Image nor a Torch Tensor.")

                # Save the image
                image_filename = f"prompt_{image_index}.png"
                image.save(os.path.join(save_path, image_filename))
                print(f"Image saved to {os.path.join(save_path, image_filename)}")
                
                return image
            else:
                print("Key 'images' not found in model output.")
                return None
    else:
        print("Model is not loaded.")
        return None

def calculate_clip_score(clip_model, clip_processor, image, prompt):
    # Process inputs
    inputs = clip_processor(text=[prompt], images=[image], return_tensors="pt", padding=True)

    # Calculate CLIP scores
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities

    return probs[0].item()

def main():
    model_1, model_2, model_3, clip_model, clip_processor = load_models()

    prompts = [
        "A busy European city street at noon with cars, cyclists, and pedestrians crossing.",
        "A quiet residential area with parked cars and trees, under a clear summer sky.",
        "An urban scene with a tram passing through a bustling market street in the fall.",
        "A snowy city street in winter, with pedestrians walking past brightly lit shops.",
        "A rainy downtown scene with neon signs reflecting on wet streets and busy sidewalks.",
        "Early morning in a suburban area, with empty roads and houses lined with flowering trees.",
        "A bridge over a river in an urban setting during sunset, with cars and buses in motion.",
        "A city park adjacent to a busy intersection, people enjoying a sunny spring afternoon.",
        "Night view of a cityscape with high-rise buildings, street lights, and sparse traffic.",
        "Sunrise over an urban landscape, with long shadows and early commuters on bicycles and scooters.",
    ]


    save_path = "/trainingData/sage/DLCV_hentci/imgs"
    os.makedirs(save_path, exist_ok=True) 

    for i, prompt in enumerate(prompts):
        image_1 = generate_image(model_1, prompt, save_path, f'{i}_stable-diffusion-v1-5')
        image_2 = generate_image(model_2, prompt, save_path, f'{i}_stable-diffusion-v1-4')
        image_3 = generate_image(model_3, prompt, save_path, f'{i}_stable-diffusion-xl-base-1.0')

        score_1 = calculate_clip_score(clip_model, clip_processor, image_1, prompt)
        score_2 = calculate_clip_score(clip_model, clip_processor, image_2, prompt)
        score_3 = calculate_clip_score(clip_model, clip_processor, image_3, prompt)

        print(f"CLIP Score for model 1, prompt {i}: {score_1}")
        print(f"CLIP Score for model 2, prompt {i}: {score_2}")
        print(f"CLIP Score for model 3, prompt {i}: {score_3}")

if __name__ == "__main__":
    main()
