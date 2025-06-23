from PIL import Image
import os
from tqdm import tqdm

# Parameters
SCALE = 4  # Upscaling factor
input_folder = "datasets/20240917-0929_AID_ranged_degradation_splits_5_clients_blur_0.2-4_noise_0-25_alpha_1.0/test/LR"  # Replace with actual folder containing low-res images
output_folder = "inference_outputs/bicubic_upsample"  # Replace with folder where upscaled images will be saved
os.makedirs(output_folder, exist_ok=True)

# Function to load an image, apply bicubic upsampling, and save the result
def process_image_bicubic(img_path, output_path, scale):
    img = Image.open(img_path).convert('RGB')
    
    # Get the original image size
    original_width, original_height = img.size
    
    # Calculate new size after upscaling
    new_width = original_width * scale
    new_height = original_height * scale
    
    # Upsample using bicubic interpolation
    upscaled_img = img.resize((new_width, new_height), Image.BICUBIC)
    
    # Save the upscaled image
    upscaled_img.save(output_path)

# Process each image in the input folder
for img_name in tqdm(os.listdir(input_folder)):
    input_img_path = os.path.join(input_folder, img_name)
    output_img_path = os.path.join(output_folder, img_name)
    
    process_image_bicubic(input_img_path, output_img_path, SCALE)

print(f"Bicubic upsampling completed. Upscaled images are saved in {output_folder}.")
