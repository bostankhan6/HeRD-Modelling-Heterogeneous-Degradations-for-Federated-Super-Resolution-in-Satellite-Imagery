from data.dataset import SR_Dataset
import torch
from tqdm import tqdm
from models import RRDBNet, network_swinir, DRCT_arch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import random
from datetime import datetime
import numpy as np
from torch.optim.lr_scheduler import StepLR
from scripts.train import train_model
from scripts.test import test_model
import os
from torch.utils.tensorboard import SummaryWriter
from data.prepare_degra_single_client_dataloader import create_centralized_dataset
import shutil
from data.prepare_datasets import prepare_dataset

def load_checkpoint(filepath, model, optimizer, scheduler):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']+1
    best_psnr = checkpoint['best_psnr']
    best_psnr_epoch = checkpoint['best_psnr_epoch']
    log_dir  = checkpoint['log_dir']
    return epoch, best_psnr, best_psnr_epoch, log_dir

SEED = 0
SCALE = 4
ONLY_TEST_Y_CHANNEL = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED) 
torch.cuda.manual_seed_all(SEED)

# HR_path_train = "datasets/AID_SR/train/HR"
# LR_path_train = "datasets/AID_SR/train/LRx2"

main_dir = "datasets/20240929-1238_AID_ranged_degradation_splits_100_clients_blur_0.2-4_noise_0-25_alpha_1.0/client_data"

HR_path_test = "datasets/AID_SR_for_degradation_splits_large_val_test/validation/HR"
LR_path_test = "datasets/20240929-1238_AID_ranged_degradation_splits_100_clients_blur_0.2-4_noise_0-25_alpha_1.0/validation/LR"

save_interval = 10 # Save sample images after every 10 epochs
                    # during training

# dataset_train = SR_Dataset(HR_path_train, LR_path_train, train_flag=True, scale=2, patch_size=96)

dataset_test = SR_Dataset(HR_path_test, LR_path_test, train_flag=False, scale=4)

# train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=2)
train_loader = create_centralized_dataset(main_dir, train_flag=True, scale=4, patch_size=192, batch_size=4)
testloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers = True)

device = "cuda" if torch.cuda.is_available() else "cpu"

timestamp = datetime.now().strftime("%Y%m%d-%H%M")

model = RRDBNet.RRDBNet(num_rrdb=10, upscale=SCALE).to(device)
#model = network_swinir.SwinIR(upscale=SCALE).to(device)
#model = DRCT_arch.DRCT(img_size=48, upscale=2, upsampler='pixelshuffle').to(device)

loss = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), 0.0002,[0.9, 0.999])

scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

epochs = 100
local_iterations = 1000

best_psnr = 0.0
best_psnr_epoch = 0

load_checkpoint_path = None

if load_checkpoint_path is not None:
      start_epoch, best_psnr, best_psnr_epoch, log_dir = load_checkpoint('path_to_checkpoint.pth', model, optimizer, scheduler)
else:
      start_epoch=0
      expermient_details = '_100_clients_blur_0.2-4_noise_0-25_alpha_1.0'
      log_dir = f"./experiments/run_{timestamp}" + "_centralized" + expermient_details
      os.makedirs(log_dir, exist_ok=True)

source_dataset_info_file = os.path.join(os.path.dirname(main_dir),'dataset_info.log')
target_dataset_info_file = os.path.join(log_dir,'dataset_info.log')
shutil.copy(source_dataset_info_file, target_dataset_info_file)

writer = SummaryWriter(log_dir=log_dir)

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

for epoch in range(start_epoch, epochs):
    #   train_model(model, train_loader, optimizer, total_iterations=100, 
    #         loss=loss, device=device, log_interval=5)
      print(f'Starting epoch: {epoch}\n')
      
      train_model(model=model, loader= train_loader, optimizer=optimizer, 
                    total_iterations=local_iterations, loss=loss, device=device, log_interval=100)
      scheduler.step()
      
      checkpoint_path = os.path.join(log_dir, f"latest_epoch_checkpoint.pth")
      #torch.save(model.state_dict(), model_checkpoint_path)
      save_checkpoint({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_psnr': best_psnr,
        'best_psnr_epoch': best_psnr_epoch,
        'log_dir': log_dir
    }, checkpoint_path)
      
      
      psnr, ssim = test_model(model, testloader, device,SCALE=SCALE)
      
      writer.add_scalar(f"psnr", psnr, epoch)
      writer.add_scalar(f"ssim", ssim, epoch)
      
      if psnr > best_psnr:
            best_psnr = psnr
            best_psnr_epoch = epoch
            # Save the best model parameters to disk
            print(f"New best psnr found in epoch {epoch}: {psnr}. Saving best model...")
            best_model_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
      else:
            print(f"No new best model found. Best PSNR till now {best_psnr} from epoch {best_psnr_epoch}.")