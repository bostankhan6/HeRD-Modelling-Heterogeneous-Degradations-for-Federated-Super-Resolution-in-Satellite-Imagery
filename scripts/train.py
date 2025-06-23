from utils.utilities import ReverseScaleToZeroToOne #ReverseScaleToZeroTo255
from torchvision.utils import save_image
from tqdm import tqdm
import random
import os


def train_model(model, loader, optimizer, total_iterations, loss, device, log_interval=10):
    
    model.train()
    iteration = 0
    #everse_transform = ReverseScaleToZeroToOne()
    
    running_loss = 0
    while iteration < total_iterations:
        for data in loader:
            if iteration >= total_iterations:
                break
            
            HR_patch, LR_patch = data
            HR_patch = HR_patch.to(device)
            LR_patch = LR_patch.to(device)
            
            optimizer.zero_grad()
            
            SR_patch = model(LR_patch)
            pixel_loss = loss(SR_patch, HR_patch)
            pixel_loss.backward()
            optimizer.step()
            running_loss += pixel_loss.item()
            iteration+=1
            
            if iteration % log_interval == 0:
                    avg_loss = running_loss / iteration
                    print(f"Iteration: {iteration}, Average Loss: {avg_loss}")
        #scheduler.step()
        
    running_avg_loss = running_loss/iteration
        
    print("Iterations:", iteration, "Average Loss:", running_avg_loss)

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