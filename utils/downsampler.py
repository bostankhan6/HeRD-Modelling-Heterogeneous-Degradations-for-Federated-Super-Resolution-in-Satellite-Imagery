import cv2
import numpy as np
import random
import os 

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def apply_blur(image, min_kernel_size=3, max_kernel_size=7):
    kernel_size = random.choice(range(min_kernel_size, max_kernel_size + 1, 2))  # Ensure kernel size is odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def downsample(image, scale_factor=4):
    height, width = image.shape[:2]
    new_height, new_width = height // scale_factor, width // scale_factor
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def add_noise(image, noise_type="gaussian", mean=0, var_range=(5, 15)):
    row, col, ch = image.shape
    if noise_type == "gaussian":
        var = random.uniform(*var_range)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy_image = image + gauss.reshape(row, col, ch)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)

def add_compression_artifacts(image, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def create_realistic_lr_image(hr_image, scale_factor=2, blur_kernel_range=(3, 9), noise_type="gaussian", noise_mean=0, noise_var_range=(5, 15), jpeg_quality=None, seed=None):
    if seed is not None:
        set_random_seed(seed)
    
    blurred_image = apply_blur(hr_image, *blur_kernel_range)
    downsampled_image = downsample(blurred_image, scale_factor)
    noisy_image = add_noise(downsampled_image, noise_type, noise_mean, noise_var_range)
    
    if jpeg_quality is not None:
        final_lr_image = add_compression_artifacts(noisy_image, jpeg_quality)
    else:
        final_lr_image = noisy_image

    return final_lr_image

# # Load your high-resolution image
# hr_image = cv2.imread('datasets/AID_SR/HR/airport_1.jpg')

# # Generate a realistic low-resolution image with optional JPEG compression
# lr_image = create_realistic_lr_image(hr_image, jpeg_quality=None, seed=42)

# # Save or display the result
# cv2.imwrite('datasets/AID_SR/airport_1.jpg', lr_image)

def process_images_in_directory(src_dir, dst_dir, scale_factor=2, blur_kernel_range=(3, 9), noise_type="gaussian", noise_mean=0, noise_var_range=(5, 15), jpeg_quality=None, seed=None):
    # Ensure destination directory exists
    os.makedirs(dst_dir, exist_ok=True)
    
    # List all .jpg files in the source directory
    image_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        # Construct full file path
        src_path = os.path.join(src_dir, image_file)
        dst_path = os.path.join(dst_dir, image_file)
        
        # Load high-resolution image
        hr_image = cv2.imread(src_path)
        
        # Check if the image was loaded successfully
        if hr_image is None:
            print(f"Error loading image: {src_path}")
            continue
        
        # Generate low-resolution image
        lr_image = create_realistic_lr_image(hr_image, scale_factor, blur_kernel_range, noise_type, noise_mean, noise_var_range, jpeg_quality, seed)
        
        # Save the processed image to the destination directory
        cv2.imwrite(dst_path, lr_image)
        print(f"Processed and saved: {dst_path}")

# Example usage
src_directory = 'datasets/AID_SR/HR'
dst_directory = 'datasets/AID_SR/LRx2'
process_images_in_directory(src_directory, dst_directory, jpeg_quality=None, seed=42)