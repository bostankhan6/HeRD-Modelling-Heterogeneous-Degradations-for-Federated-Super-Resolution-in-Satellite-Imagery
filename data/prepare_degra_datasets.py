
import os
from data.dataset import SR_Dataset
from torch.utils.data import DataLoader

def create_datasets_from_clients(main_dir, LR_type, train_flag=True, scale=4, patch_size=192, batch_size=4):
    """
    Create a list of SR_Dataset objects, each corresponding to a different client.
    
    :param main_dir: The main directory containing client folders.
    :param train_flag: Flag to indicate if the dataset is for training or testing.
    :param scale: The scale factor for downsampling.
    :param patch_size: The size of the patch to be extracted.
    :return: A list of SR_Dataset objects, one for each client.
    """
    client_dirs = [os.path.join(main_dir, client) for client in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, client))]
    #datasets = []
    loaders = []
    
    for client_dir in client_dirs:
        hr_path = os.path.join(client_dir, 'HR')
        lr_path = os.path.join(client_dir, 'LR') + "_" + LR_type
        
        if os.path.exists(hr_path) and os.path.exists(lr_path):
            dataset = SR_Dataset(HR_path=hr_path, LR_path=lr_path, train_flag=train_flag, scale=scale, patch_size=patch_size)
            loader = DataLoader(dataset, batch_size, shuffle=True)
            loaders.append(loader)
        else:
            print(f"HR or LR folder missing in {client_dir}")
        
    
    return loaders