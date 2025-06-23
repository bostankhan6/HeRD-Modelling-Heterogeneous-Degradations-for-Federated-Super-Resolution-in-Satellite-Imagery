import flwr as fl

from collections import OrderedDict
from typing import Dict, Tuple
import os
import torch
from flwr.common import NDArrays, Scalar
import random
import numpy as np
from models.RRDBNet import RRDBNet
#from model_AsConvSR import AsConvSR
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from scripts.train import train_model
from scripts.test import test_model
import copy
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, #valloader,
                 device) -> None:
        super().__init__()

        self.cid = cid
        self.trainloader = trainloader
        #self.valloader = valloader
        self.device = device #"cuda" if torch.cuda.is_available() else "cpu"
        self.model = model

        self.optimizer = torch.optim.Adam(self.model.parameters(), 0.0002,[0.9, 0.999])       
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.loss = nn.L1Loss()
        self.save_interval = 1
        
    def set_parameters(self, parameters):
        """With the model parameters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        # now replace the parameters
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and convert them to a list of
        NumPy arrays. The server doesn't work with PyTorch/TF/etc."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # Read values from config
        server_round = config["server_round"]
        local_iterations = config["local_iterations"]
        log_interval = config["log_interval"]
        log_dir = config["log_dir"]

        # if server_round-1 == best_psnr_round:
        #         client_model_path = os.path.join(log_dir, f"client_{self.cid}_best_model.pth")
        #         torch.save(self.model.state_dict(), client_model_path)
        #         print(f"Saving client_{self.cid} model as best for server round {best_psnr_round}.")

        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        
        # do local training  -------------------------------------------------------------- Essentially the same as in the centralised example above (but now using the client's data instead of the whole dataset)
        train_model(model=self.model, loader= self.trainloader, optimizer=self.optimizer, 
                    total_iterations=local_iterations, loss=self.loss, device=self.device, log_interval=log_interval)
        self.scheduler.step()
        # return the model parameters to the server as well as extra info (number of training examples in this case)
        
        client_model_path = os.path.join(log_dir, f"client_{self.cid}_latest_model.pth")
        #torch.save(self.model.state_dict(), client_model_path)
        print(f"Saving client_{self.cid} model for server round {server_round}.")
       
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        #self.set_parameters(parameters)
        
        #psnr, ssim = test_model(self.model, self.valloader, self.device)
        
        # send statistics back to the server
        #return ssim, len(self.valloader), {"psnr": psnr, "ssim": ssim} 
        return 0.0, 1, {}
    
def generate_client_fn(trainloaders, #valloaders,
                       device, model):
    def client_fn(cid: str):
        """Returns a FlowerClient containing the cid-th data partition"""
        #model = RRDBNet(upscale=2).to(device=device)
        
        client_model = copy.deepcopy(model).to(device=device)
        #client_model = RRDBNet(num_rrdb=10,upscale=4).to(device=device)
        return FlowerClient(cid, client_model,
            trainloader=trainloaders[int(cid)], #valloader=valloaders[int(cid)],
            device=device
        )

    return client_fn