from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from utils.utilities import ScaleToMinusOneToOne
import random

class SR_Dataset(Dataset):
    def __init__(self, HR_path, LR_path, train_flag=True, scale=4, patch_size=192 ):
        #LR_type: one of 'none', 'blur', 'noise' or 'both'. Default is 'both'
        super(SR_Dataset, self).__init__()
        self.scale = scale
        self.HR_path = HR_path
        self.LR_path = LR_path
        self.patch_size = patch_size  # Patch size for HR image
        self.HR_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            ScaleToMinusOneToOne() 
        ])
        self.LR_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            ScaleToMinusOneToOne() 
        ])
        self.file_pairs = self.create_file_pairs()
        self.train_flag = train_flag 
        

    def create_file_pairs(self):
        hr_files = [file for file in os.listdir(self.HR_path) if file.endswith(('.png', '.jpg'))]
        
        file_pairs = []
        for hr_file in hr_files:
            file_name, ext = os.path.splitext(hr_file)
            HR_file_path = os.path.join(self.HR_path, hr_file)
            LR_file_path_png = os.path.join(self.LR_path, file_name + '.png')
            LR_file_path_jpg = os.path.join(self.LR_path, file_name + '.jpg')
            
            if os.path.exists(LR_file_path_png):
                file_pairs.append((HR_file_path, LR_file_path_png))
            elif os.path.exists(LR_file_path_jpg):
                file_pairs.append((HR_file_path, LR_file_path_jpg))
            else:
                print(f"LR 'x{self.scale}' file does not exist for {file_name} in {self.LR_path} directory.")
                
        return file_pairs
        
    def __len__(self):
        return len(self.file_pairs) 


    def __getitem__(self, idx):
        HR_path, LR_path = self.file_pairs[idx]
        
        HR_image = Image.open(HR_path).convert('RGB')
        LR_image = Image.open(LR_path).convert('RGB')

        if self.train_flag:
            # Randomly extract a patch from HR image
            hr_width, hr_height = HR_image.size
            left = random.randint(0, hr_width - self.patch_size)
            top = random.randint(0, hr_height - self.patch_size)
            HR_image = HR_image.crop((left, top, left + self.patch_size, top + self.patch_size))

            # Corresponding patch from LR image (self.scale times smaller)
            LR_image = LR_image.crop((left // self.scale, top // self.scale, (left + self.patch_size) // self.scale, (top + self.patch_size) // self.scale))

            # Data augmentation
            if random.random() > 0.5:
                HR_image = HR_image.transpose(Image.FLIP_LEFT_RIGHT)
                LR_image = LR_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            rotation_angle = random.choice([0, 90, 180, 270])
            if rotation_angle != 0:
                HR_image = HR_image.rotate(rotation_angle)
                LR_image = LR_image.rotate(rotation_angle)
            
        HR_image = self.HR_transform(HR_image)
        LR_image = self.LR_transform(LR_image)

        return HR_image, LR_image

    
# dataset = SR_Dataset('dataset/HR', 'dataset/LR', patch_size=256)

# # Access a single HR patch and its corresponding LR patch
# HR_patch, LR_patch = dataset[0]  # Access the first pair, change the index as needed

# # Convert tensors to PIL images for visualization
# to_pil = transforms.ToPILImage()
# HR_patch_img = to_pil(HR_patch)
# LR_patch_img = to_pil(LR_patch)

# # Display HR and LR patches
# plt.figure(figsize=(8, 4))

# plt.subplot(1, 2, 1)
# plt.title('HR Patch')
# plt.imshow(HR_patch_img)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('LR Patch')
# plt.imshow(LR_patch_img)
# plt.axis('off')

# plt.tight_layout()
# plt.show()