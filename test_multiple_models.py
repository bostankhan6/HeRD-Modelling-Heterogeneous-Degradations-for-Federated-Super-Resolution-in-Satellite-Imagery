from data.dataset import SR_Dataset
from torch.utils.data import DataLoader
from models import RRDBNet
import torch
from scripts.test import test_model
from datetime import datetime

# Paths and configuration
HR_path_test = "datasets/AID_SR_for_degradation_splits_large_val_test/test/HR"
LR_path_test = "datasets/20250124-1117_AID_ranged_degradation_splits_2_clients_blur_3-4_noise_10-25_alpha_1.0/test/LR_both"
SCALE = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare test dataset and dataloader
dataset_test = SR_Dataset(HR_path_test, LR_path_test, train_flag=False, scale=SCALE)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                         pin_memory=True, persistent_workers=True)

# Load model
model = RRDBNet.RRDBNet(num_rrdb=10, upscale=SCALE).to(device)

model_path = "experiments/run_20250121-1605_federated_10_clients_blur_1_noise_1_alpha_1.0_FedAvg/best_model.pth"
state_dict = torch.load(model_path)
if "epoch" in state_dict:
    model.load_state_dict(state_dict["model_state_dict"])
else:
    model.load_state_dict(state_dict)

# Test the model
psnr, ssim = test_model(model, test_loader, device, SCALE=SCALE)

# Save results to a log file
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = f"test_logs/log_{timestamp}.txt"

# Ensure the log directory exists
import os
os.makedirs("test_logs", exist_ok=True)

with open(log_file, "w") as log:
    log.write("Model Testing Log\n")
    log.write("====================\n")
    log.write(f"Test Date and Time: {timestamp}\n")
    log.write(f"Model Path: {model_path}\n")
    log.write(f"Testing Dataset: {HR_path_test}\n")
    log.write(f"Scale Factor: {SCALE}\n")
    log.write("====================\n")
    log.write(f"PSNR: {psnr:.4f}\n")
    log.write(f"SSIM: {ssim:.4f}\n")

print(f"Testing completed. Results saved to {log_file}.")
