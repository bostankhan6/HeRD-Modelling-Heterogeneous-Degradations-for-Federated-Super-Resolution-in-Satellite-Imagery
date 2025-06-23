import numpy as np
import cv2
import os
from PIL import Image
from scipy.ndimage import rotate
from datetime import datetime

np.random.seed(42)

num_clients = 10  # 5 clients in previous scenario
scale_factor = 4  # 4 in in previous scenario
kernel_size = 21

# --------------------------------------------------------------------------------
# We define the ranges for sigma and noise using Dirichlet distribution
# --------------------------------------------------------------------------------
max_sigma = 4
min_sigma = 0.2
max_noise_level = 25
min_noise_level = 0
alpha = 1.0  # Dirichlet distribution parameter

dataset_dir = 'datasets/AID_HR_images/train'
validation_hr_dir = 'datasets/AID_HR_images/validation'
test_hr_dir = 'datasets/AID_HR_images/test'

def apply_isotropic_gaussian_blur(image, sigma):
    """Apply isotropic Gaussian blur."""
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image

def generate_anisotropic_gaussian_kernel(size, lambda1, lambda2, theta):
    """Generates an anisotropic Gaussian kernel."""
    # Create a grid of (x, y) coordinates
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)

    # Rotate the grid by the specified angle theta
    theta_rad = theta
    xx_rot = np.cos(theta_rad) * xx + np.sin(theta_rad) * yy
    yy_rot = -np.sin(theta_rad) * xx + np.cos(theta_rad) * yy

    # Create the anisotropic Gaussian using the rotated grid
    kernel = np.exp(-0.5 * ((xx_rot ** 2) / lambda1 ** 2 + (yy_rot ** 2) / lambda2 ** 2))

    # Normalize the kernel to ensure the sum of all elements equals 1
    kernel /= np.sum(kernel)
    return kernel

def apply_anisotropic_gaussian_blur(image, lambda1, lambda2, theta):
    """Applies anisotropic Gaussian blur to an image."""
    kernel = generate_anisotropic_gaussian_kernel(kernel_size, lambda1, lambda2, theta)
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

def add_noise(image, noise_level):
    """Add Gaussian noise."""
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

# --------------------------------------------------------------------------------
# NEW function: degrade_image_variant
# --------------------------------------------------------------------------------
def degrade_image_variant(
    image,
    variant,
    blur_settings=None,   # dictionary for blur settings (e.g., {type:'isotropic' or 'anisotropic', sigma_range, lambda1_range, etc.})
    noise_settings=None   # dictionary for noise settings (e.g., {noise_level_range: (min, max)})
):
    """
    Degrade the image according to the specified variant:
      - 'none': no blur, no noise
      - 'blur': blur only
      - 'noise': noise only
      - 'both': blur + noise

    For blur, we will apply anisotropic (as indicated in your code), 
    but feel free to switch to isotropic if you prefer.
    """
    degraded = image.copy()

    # Handle blur
    if variant in ['blur', 'both']:
        # Decide if you want anisotropic or isotropic:
        # For example, let's do anisotropic:
        if blur_settings['degradation_type'] == 'isotropic':
            sigma = np.random.uniform(*blur_settings['sigma_range'])
            degraded = apply_isotropic_gaussian_blur(degraded, sigma)
        else:  # 'anisotropic'
            lambda1 = np.random.uniform(*blur_settings['lambda1_range'])
            lambda2 = np.random.uniform(*blur_settings['lambda2_range'])
            theta   = np.random.uniform(*blur_settings['theta_range'])
            degraded = apply_anisotropic_gaussian_blur(degraded, lambda1, lambda2, theta)

    # Handle noise
    if variant in ['noise', 'both']:
        noise_level = np.random.uniform(*noise_settings['noise_level_range'])
        degraded = add_noise(degraded, noise_level)

    return degraded

def downsample(image, scale_factor):
    """Downsample the image by the specified scale factor."""
    h, w = image.shape[:2]
    downsampled_image = cv2.resize(image, (int(w/scale_factor), int(h/scale_factor)), interpolation=cv2.INTER_CUBIC)
    return downsampled_image

sigma_ranges = np.cumsum(np.random.dirichlet([alpha] * num_clients)) * (max_sigma - min_sigma) + min_sigma
noise_ranges = np.cumsum(np.random.dirichlet([alpha] * num_clients)) * (max_noise_level - min_noise_level) + min_noise_level

sigma_ranges = np.concatenate([[min_sigma], sigma_ranges])
noise_ranges = np.concatenate([[min_noise_level], noise_ranges])

# --------------------------------------------------------------------------------
# Create per-client degradation settings
# Defaulting to anisotropic blur in your original code, but can be set to 'isotropic' if you wish
# --------------------------------------------------------------------------------
degradation_settings = []
for i in range(num_clients):
    degradation_type = 'anisotropic'  # or 'isotropic' if you prefer
    settings = {
        'degradation_type': degradation_type,
        'sigma_range': (sigma_ranges[i], sigma_ranges[i + 1]) if degradation_type == 'isotropic' else None,
        'lambda1_range': (sigma_ranges[i], sigma_ranges[i + 1]) if degradation_type == 'anisotropic' else None,
        'lambda2_range': (sigma_ranges[i], sigma_ranges[i + 1]) if degradation_type == 'anisotropic' else None,
        'theta_range': (0, np.pi) if degradation_type == 'anisotropic' else None,
        'noise_level_range': (noise_ranges[i], noise_ranges[i + 1])
    }
    degradation_settings.append(settings)

# --------------------------------------------------------------------------------
# Modify the create_client_datasets function to produce 4 LR subfolders:
# 1) LR_none
# 2) LR_blur
# 3) LR_noise
# 4) LR_both
# --------------------------------------------------------------------------------
def create_client_datasets(
    aid_dataset_dir, 
    output_dir, 
    degradation_settings, 
    num_clients=5, 
    scale_factor=4, 
    split_validation=False, 
    val_split=0.1
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load all image paths from the AID dataset
    all_image_paths = [
        os.path.join(aid_dataset_dir, img) 
        for img in os.listdir(aid_dataset_dir) 
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Shuffle the images
    np.random.shuffle(all_image_paths)

    # If validation split is required, split the data
    if split_validation:
        split_index = int((1 - val_split) * len(all_image_paths))
        training_paths = all_image_paths[:split_index]
        validation_paths = all_image_paths[split_index:]
    else:
        training_paths = all_image_paths
        validation_paths = None

    # Split the training data into num_clients subsets
    split_size = len(training_paths) // num_clients
    client_datasets = []

    for i in range(num_clients):
        client_image_paths = training_paths[i * split_size:(i + 1) * split_size]
        client_datasets.append(client_image_paths)

        # Create client-specific directories
        client_dir = os.path.join(output_dir, f'client_{i}')
        os.makedirs(client_dir, exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'HR'), exist_ok=True)

        # Create four LR subdirectories: LR_none, LR_blur, LR_noise, LR_both
        os.makedirs(os.path.join(client_dir, 'LR_none'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'LR_blur'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'LR_noise'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'LR_both'), exist_ok=True)

        # Process each image for the client
        for img_path in client_image_paths:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)

            # --------------------
            # Save the HR image
            # --------------------
            img_filename = os.path.basename(img_path)
            hr_image_pil = Image.fromarray(img_np)
            hr_image_pil.save(os.path.join(client_dir, 'HR', img_filename))

            # --------------------
            # Create the four LR variants
            # --------------------
            blur_settings = degradation_settings[i]  # the client's blur info
            noise_settings = degradation_settings[i] # the client's noise info

            for variant in ['none', 'blur', 'noise', 'both']:
                degraded_img = degrade_image_variant(
                    img_np,
                    variant,
                    blur_settings=blur_settings,
                    noise_settings=noise_settings
                )
                lr_image = downsample(degraded_img, scale_factor)
                lr_image_pil = Image.fromarray(lr_image)

                # Choose correct subfolder name
                if variant == 'none':
                    folder_name = 'LR_none'
                elif variant == 'blur':
                    folder_name = 'LR_blur'
                elif variant == 'noise':
                    folder_name = 'LR_noise'
                else:  # 'both'
                    folder_name = 'LR_both'

                lr_image_pil.save(os.path.join(client_dir, folder_name, img_filename))

        # Now, create validation images for the client if validation data is split
        if split_validation and validation_paths is not None:
            val_dir = os.path.join(client_dir, 'validation')
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(os.path.join(val_dir, 'HR'), exist_ok=True)

            # Again, create the four LR subdirs inside validation
            os.makedirs(os.path.join(val_dir, 'LR_none'), exist_ok=True)
            os.makedirs(os.path.join(val_dir, 'LR_blur'), exist_ok=True)
            os.makedirs(os.path.join(val_dir, 'LR_noise'), exist_ok=True)
            os.makedirs(os.path.join(val_dir, 'LR_both'), exist_ok=True)

            # Split the validation data for each client
            val_subset_size = len(validation_paths) // num_clients
            client_validation_paths = validation_paths[i * val_subset_size : (i + 1) * val_subset_size]

            for val_img_path in client_validation_paths:
                val_img = Image.open(val_img_path).convert('RGB')
                val_img_np = np.array(val_img)
                val_img_filename = os.path.basename(val_img_path)

                # Save HR
                hr_val_image_pil = Image.fromarray(val_img_np)
                hr_val_image_pil.save(os.path.join(val_dir, 'HR', val_img_filename))

                # Create the four LR variants for validation
                blur_settings = degradation_settings[i]
                noise_settings = degradation_settings[i]

                for variant in ['none', 'blur', 'noise', 'both']:
                    degraded_val_img = degrade_image_variant(
                        val_img_np,
                        variant,
                        blur_settings=blur_settings,
                        noise_settings=noise_settings
                    )
                    lr_val_image = downsample(degraded_val_img, scale_factor)
                    lr_val_image_pil = Image.fromarray(lr_val_image)

                    if variant == 'none':
                        folder_name = 'LR_none'
                    elif variant == 'blur':
                        folder_name = 'LR_blur'
                    elif variant == 'noise':
                        folder_name = 'LR_noise'
                    else:  # 'both'
                        folder_name = 'LR_both'

                    lr_val_image_pil.save(os.path.join(val_dir, folder_name, val_img_filename))



def create_lr_images_for_validation_test(
    hr_dir,
    output_lr_dir,
    degradation_settings,
    scale_factor=4,
    include_variants=['none', 'blur', 'noise', 'both']
):
    """
    include_variants is a list that decides which variants are created:
      - For the new federated-validation set: ['none', 'blur', 'noise', 'both']
      - For the test set: ['both'] only
    """
    if not os.path.exists(output_lr_dir):
        os.makedirs(output_lr_dir)

    # If we need subfolders for each variant, create them.
    # If include_variants has multiple items, we create subfolders. 
    # If it has only 'both', we only create that subfolder.
    for variant in include_variants:
        variant_dir = os.path.join(output_lr_dir, f'LR_{variant}')
        os.makedirs(variant_dir, exist_ok=True)

    # Load all image paths from the HR directory
    all_image_paths = [
        os.path.join(hr_dir, img) 
        for img in os.listdir(hr_dir) 
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for img_path in all_image_paths:
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        img_filename = os.path.basename(img_path)

        # Randomly select a degradation setting from existing client settings
        selected_degradation = degradation_settings[np.random.randint(len(degradation_settings))]

        for variant in include_variants:
            # Degrade image
            degraded_img = degrade_image_variant(
                img_np,
                variant,
                blur_settings=selected_degradation,
                noise_settings=selected_degradation
            )
            # Downsample
            lr_image = downsample(degraded_img, scale_factor)
            lr_image_pil = Image.fromarray(lr_image)

            # Save in the proper subfolder
            lr_image_pil.save(
                os.path.join(output_lr_dir, f'LR_{variant}', img_filename)
            )

def save_log_file(
    output_dir,
    num_clients,
    max_sigma,
    min_sigma,
    max_noise_level,
    min_noise_level,
    alpha,
    degradation_settings
):
    log_file_path = os.path.join(output_dir, "dataset_info.log")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Number of clients: {num_clients}\n")
        log_file.write(f"Scale factor: {scale_factor}\n")
        log_file.write(f"Kernel size: {kernel_size}\n")
        log_file.write(f"Max sigma: {max_sigma}\n")
        log_file.write(f"Min sigma: {min_sigma}\n")
        log_file.write(f"Max noise level: {max_noise_level}\n")
        log_file.write(f"Min noise level: {min_noise_level}\n")
        log_file.write(f"Alpha (Dirichlet distribution parameter): {alpha}\n")
        log_file.write(f"Degradation settings per client:\n")
        for i, settings in enumerate(degradation_settings):
            log_file.write(f"  Client {i + 1}:\n")
            log_file.write(f"    Degradation type: {settings['degradation_type']}\n")
            if settings['degradation_type'] == 'isotropic':
                log_file.write(f"    Sigma range: {settings['sigma_range']}\n")
            elif settings['degradation_type'] == 'anisotropic':
                log_file.write(f"    Lambda1 range: {settings['lambda1_range']}\n")
                log_file.write(f"    Lambda2 range: {settings['lambda2_range']}\n")
                log_file.write(f"    Theta range: {settings['theta_range']}\n")
            log_file.write(f"    Noise level range: {settings['noise_level_range']}\n")

timestamp = datetime.now().strftime("%Y%m%d-%H%M")

output_dir = f'datasets/{timestamp}_AID_ranged_degradation_splits_{num_clients}_clients_blur_{min_sigma}-{max_sigma}_noise_{min_noise_level}-{max_noise_level}_alpha_{alpha}'
client_dir = os.path.join(output_dir, 'client_data')

# 1) Create client training+validation with 4 variants each (none, blur, noise, both)
create_client_datasets(
    dataset_dir,
    client_dir,
    degradation_settings,
    num_clients=num_clients,
    scale_factor=scale_factor,
    split_validation=True,
    val_split=0.1
)

# 2) Save a log file
save_log_file(
    output_dir,
    num_clients,
    max_sigma,
    min_sigma,
    max_noise_level,
    min_noise_level,
    alpha,
    degradation_settings
)

# 3) Create the federated-validation set (the global validation set used by the FL model)
#    with 4 LR variants: none, blur, noise, both
validation_output_dir = os.path.join(output_dir, 'federated_validation')
create_lr_images_for_validation_test(
    validation_hr_dir,
    validation_output_dir,
    degradation_settings,
    scale_factor=scale_factor,
    include_variants=['none', 'blur', 'noise', 'both']  # produce all 4
)

# 4) Create the test set
test_output_dir = os.path.join(output_dir, 'test')
create_lr_images_for_validation_test(
    test_hr_dir,
    test_output_dir,
    degradation_settings,
    scale_factor=scale_factor,
    include_variants=['none', 'blur', 'noise', 'both'] # produce all 4 but only 'both' is used
)
