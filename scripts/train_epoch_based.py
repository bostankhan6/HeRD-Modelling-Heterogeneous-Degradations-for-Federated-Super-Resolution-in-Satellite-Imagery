from utils.utilities import ReverseScaleToZeroToOne #ReverseScaleToZeroTo255
from torchvision.utils import save_image
from tqdm import tqdm
import random
import os


def train_model(model, loader, optimizer, scheduler, epochs, loss, device, save_interval=2):
    
    model.train()
    reverse_transform = ReverseScaleToZeroToOne()
    for epoch in range(epochs):
        epoch_avg_loss = 0
        for data in tqdm(loader):
            HR_patch, LR_patch = data
            HR_patch = HR_patch.to(device)
            LR_patch = LR_patch.to(device)
            
            optimizer.zero_grad()
            
            SR_patch = model(LR_patch)
            pixel_loss = loss(SR_patch, HR_patch)
            pixel_loss.backward()
            optimizer.step()
            epoch_avg_loss += pixel_loss.item()
        
        scheduler.step()
            
        epoch_avg_loss = epoch_avg_loss/len(loader)
            
        print("Epoch:", epoch, "Loss:", epoch_avg_loss)

        # Save images after every 'save_interval' epochs
        # if epoch % save_interval == 0:
            
        #     for i in range(2):  # Save only 2 images
        #         # Randomly select an index from SR_patch batch
        #         random_index = random.randint(0, len(SR_patch) - 1)
        #         selected_image_sr = SR_patch[random_index].detach().cpu()  # Detach and move to CPU
        #         selected_image_hr = HR_patch[random_index].detach().cpu()
                
        #         selected_image_sr = reverse_transform(selected_image_sr)
        #         selected_image_hr = reverse_transform(selected_image_hr)
                
        #         # Save the image using torchvision's save_image function
        #         save_image(selected_image_sr, os.path.join('runs/exp1', f'{epoch}_{i}_sr.png'))
        #         save_image(selected_image_hr, os.path.join('runs/exp1', f'{epoch}_{i}_hr.png'))
                
        #         # print("Images saved successfully.")