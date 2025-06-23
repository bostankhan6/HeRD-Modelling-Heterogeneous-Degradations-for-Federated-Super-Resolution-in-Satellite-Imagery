import numpy as np
import cv2
import os
from PIL import Image
from scipy.ndimage import rotate

def apply_isotropic_gaussian_blur(image, sigma):
    """Apply isotropic Gaussian blur."""
    kernel_size = 21
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image

def apply_anisotropic_gaussian_blur(image, lambda1, lambda2, theta):
    """Apply anisotropic Gaussian blur."""
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    covariance_matrix = rotation_matrix @ np.diag([lambda1, lambda2]) @ rotation_matrix.T
    kernel_size = 21
    kernel = cv2.getGaussianKernel(kernel_size, np.sqrt(np.mean([lambda1, lambda2])))
    kernel = np.outer(kernel, kernel.T)
    
    kernel_rotated = rotate(kernel, angle=np.degrees(theta), reshape=False)
    blurred_image = cv2.filter2D(image, -1, kernel_rotated)
    return blurred_image

def add_noise(image, noise_level):
    """Add Gaussian noise."""
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def degrade_image(image, degradation_settings):
    """Degrade the image based on the specified settings."""
    if degradation_settings['degradation_type'] == 'isotropic':
        degraded_image = apply_isotropic_gaussian_blur(image, degradation_settings['sigma'])
    elif degradation_settings['degradation_type'] == 'anisotropic':
        degraded_image = apply_anisotropic_gaussian_blur(image, degradation_settings['lambda1'], degradation_settings['lambda2'], degradation_settings['theta'])
    else:
        raise ValueError("Unknown degradation type.")
    
    degraded_image = add_noise(degraded_image, degradation_settings['noise_level'])
    
    return degraded_image

def downsample(image, scale_factor):
    """Downsample the image by the specified scale factor."""
    h, w = image.shape[:2]
    downsampled_image = cv2.resize(image, (int(w/scale_factor), int(h/scale_factor)), interpolation=cv2.INTER_CUBIC)
    return downsampled_image

sigma = 10
# Define random degradation settings
def get_random_degradation_settings():
    degradation_type = np.random.choice(['isotropic', 'anisotropic'])
    settings = {
        'degradation_type': degradation_type,
        'sigma': np.random.uniform(0.2, sigma) if degradation_type == 'isotropic' else None,
        'lambda1': np.random.uniform(0.2, sigma) if degradation_type == 'anisotropic' else None,
        'lambda2': np.random.uniform(0.2, sigma) if degradation_type == 'anisotropic' else None,
        'theta': np.random.uniform(0, np.pi) if degradation_type == 'anisotropic' else None,
        'noise_level': np.random.uniform(0, 25)
    }
    return settings

def create_lr_images_for_client(source_dir, lr_dir, scale_factor=4):
    # Create LR directory if it doesn't exist
     
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
    
    # Load all image paths from the source directory
    all_image_paths = [os.path.join(source_dir, img) for img in os.listdir(source_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_path in all_image_paths:
        img = Image.open(img_path)
        img_np = np.array(img)

        # Apply random degradations and downsample each image
        degradation_settings = get_random_degradation_settings()

        # Degrade the image
        degraded_img = degrade_image(img_np, degradation_settings)
        
        # Downsample to create the LR image
        lr_image = downsample(degraded_img, scale_factor)
        
        # Save the LR image
        img_filename = os.path.basename(img_path)
        lr_image_pil = Image.fromarray(lr_image)
        lr_image_pil.save(os.path.join(lr_dir, img_filename))

# Example usage:
source_dir = 'datasets/AID_SR/test/HR'
lr_dir = 'datasets/AID_SR/test/LRx4_multidegra'
create_lr_images_for_client(source_dir, lr_dir, scale_factor=4)