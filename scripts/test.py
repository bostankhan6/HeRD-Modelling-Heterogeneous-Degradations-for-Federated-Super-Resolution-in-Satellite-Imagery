import torch
from tqdm import tqdm
from utils.utilities import build_iqa_model, calculate_psnr_pt, calculate_ssim_pt
from utils.utilities import  ReverseScaleToZeroToOne

def test_model(model, loader, device, SCALE=4, 
         ONLY_TEST_Y_CHANNEL=False): 
    
    psnr_model, ssim_model = build_iqa_model(
        SCALE,
        True,#ONLY_TEST_Y_CHANNEL,
        device,
    )       
       
    model.eval()
    avg_psnr = 0
    avg_ssim = 0
    with torch.no_grad():
        for data in tqdm(loader):
            HR_image, LR_image = data
            HR_image = HR_image.to(device)
            LR_image = LR_image.to(device)
            
            SR_image = model(LR_image)
            
            SR_image = ReverseScaleToZeroToOne()(SR_image)
            HR_image = ReverseScaleToZeroToOne()(HR_image)
            
            psnr = calculate_psnr_pt(SR_image, HR_image, SCALE, True)
            #psnr = psnr_model(SR_image, HR_image)
            #ssim = ssim_model(SR_image, HR_image)
            ssim = calculate_ssim_pt(SR_image, HR_image, SCALE, test_y_channel=True)

            # print(psnr)
            avg_psnr += psnr.item()
            avg_ssim += ssim.item()
            
    avg_psnr = avg_psnr/len(loader)
    avg_ssim = avg_ssim/len(loader)
    
    print(f'Average PSNR: {avg_psnr:.2f}')
    print(f'Average SSIM: {avg_ssim:.4f}')

    return avg_psnr, avg_ssim