import flwr as fl
from flower.evaluation_strategy import get_evalulate_fn, weighted_average
from flower.flower_client import generate_client_fn
from data.non_iid_dataset import SR_Dataset as class_sr_dataset
from data.dataset import SR_Dataset
import matplotlib.pyplot as plt
from data.prepare_datasets import prepare_dataset
import torch
from models import RRDBNet, network_swinir, DRCT_arch
from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import glob
import os
from torch.utils.tensorboard import SummaryWriter
import copy
from datetime import datetime
from torch.utils.data import DataLoader
# from data.prepare_noniid_datasets import prepare_noniid_dataset
from data.prepare_degra_datasets import create_datasets_from_clients
import random
import shutil
import os

from flwr.common import (
    ndarrays_to_parameters
)

# Set seeds for reproducibility
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set the seed
SEED = 42
set_seed(SEED)
               

class SaveModelStrategy(fl.server.strategy.FedProx):
    def __init__(self, *args, log_dir="./logs", **kwargs):
        super().__init__(*args, **kwargs)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ):
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model
            model_checkpoint_path = os.path.join(self.log_dir, f"latest_model.pth") #_round_{server_round}.pth")
            torch.save(model.state_dict(), model_checkpoint_path)

        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round: int, parameters):
        loss, metrics = super().evaluate(server_round, parameters)
        for metric, value in metrics.items():
                self.writer.add_scalar(f"{metric}", value, server_round)
        return loss, metrics

def load_model(model,path):
    list_of_files = [fname for fname in glob.glob(os.path.join(path, "model_round_*"))]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    model.load_state_dict(state_dict)
    state_dict_ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]
    parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)
    return parameters

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_iterations": 1000, #if server_round < 2 else 2,  #
        "log_interval": 100,
        "log_dir" : log_dir
    }
    return config

device = "cuda" if torch.cuda.is_available() else "cpu"

train_LR_type = 'both'

main_dir = "datasets/20250116-1049_AID_ranged_degradation_splits_5_clients_blur_0.2-4_noise_0-25_alpha_0.1/client_data"

HR_path_val = "datasets/AID_SR_for_degradation_splits_large_val_test/validation/HR"

LR_path_val = "datasets/20250116-1049_AID_ranged_degradation_splits_5_clients_blur_0.2-4_noise_0-25_alpha_0.1/federated_validation/LR"

LR_path_val = LR_path_val + '_' + train_LR_type

NUM_CLIENTS = 5
SCALE = 4

dataset_val = SR_Dataset(HR_path_val, LR_path_val, train_flag=False, scale=SCALE, patch_size=192)

trainloaders = create_datasets_from_clients(main_dir, LR_type = train_LR_type, train_flag=True, scale=SCALE, patch_size=192, batch_size=4)

valloader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers = True)

model = RRDBNet.RRDBNet(num_rrdb=10,upscale=SCALE)

model_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])

timestamp = datetime.now().strftime("%Y%m%d-%H%M")
expermient_details = '_5_clients_blur_1_noise_1_alpha_0.1_FedProx1.0'
log_dir = f"./experiments/run_{timestamp}" + "_federated" + expermient_details
os.makedirs(log_dir, exist_ok=True)

if os.path.exists(os.path.join(os.path.dirname(main_dir),'dataset_info.log')):
    source_dataset_info_file = os.path.join(os.path.dirname(main_dir),'dataset_info.log')
    target_dataset_info_file = os.path.join(log_dir,'dataset_info.log')
    shutil.copy(source_dataset_info_file, target_dataset_info_file)

strategy = SaveModelStrategy(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0,  # Sample 100% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 2 clients for training
    min_evaluate_clients=0,  # Never sample less than 2 clients for evaluation
    min_available_clients= 5,  # Wait until all 2 clients are available
    evaluate_fn=get_evalulate_fn(valloader,device, copy.deepcopy(model).to(device=device), log_dir),
    on_fit_config_fn=fit_config,
    log_dir=log_dir,
    proximal_mu=1.0, 
    #initial_parameters = model_parameters
)

client_fn_callback = generate_client_fn(trainloaders, #valloaders,
                                        device, model=model)

my_client_resources = {'num_cpus': 1, 'num_gpus': 0.20}#1.0/NUM_CLIENTS}

fl.common.logger.configure(identifier="RRDB_experiment", filename=os.path.join(log_dir,"experiment_log.txt"))

history = fl.simulation.start_simulation(
    client_fn=client_fn_callback,  # a callback to construct a client
    num_clients=NUM_CLIENTS,  # total number of clients in the experiment
    config=fl.server.ServerConfig(num_rounds=100),  # let's run for 10 rounds
    strategy=strategy,  # the strategy that will orchestrate the whole FL pipeline
    ray_init_args = {'num_gpus': 1},
    client_resources = my_client_resources,    
) 

print(f"{history.metrics_centralized = }")

psnr_centralised = history.metrics_centralized["psnr"]
round = [data[0] for data in psnr_centralised]
psnr = [data[1] for data in psnr_centralised]
plt.plot(round, psnr)
plt.grid()
plt.ylabel("psnr (%)")
plt.xlabel("Round")
plt.title("PSNR over Rounds")
plt.show()