from data.dataset import SR_Dataset
from torch.utils.data import DataLoader
from models import RRDBNet
import torch
from scripts.test import test_model

HR_path_test = "datasets/AID_SR_for_degradation_splits_large_val_test/test/HR"
LR_path_test = "datasets/20250124-1117_AID_ranged_degradation_splits_2_clients_blur_3-4_noise_10-25_alpha_1.0/test/LR_both"
SCALE = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
 
dataset_test = SR_Dataset(HR_path_test, LR_path_test, train_flag=False, scale=SCALE)

test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers = True)

model = RRDBNet.RRDBNet(num_rrdb=10,upscale=SCALE).to(device)

model_path = "experiments/run_20250122-0945_federated_5_clients_blur_1_noise_1_alpha_0.1_FedProx1.0/best_model.pth"

state_dict = torch.load(model_path)
if "epoch" in state_dict:
    model.load_state_dict(state_dict["model_state_dict"])
else:
    model.load_state_dict(state_dict)

psnr, ssim = test_model(model, test_loader, device,SCALE=SCALE)