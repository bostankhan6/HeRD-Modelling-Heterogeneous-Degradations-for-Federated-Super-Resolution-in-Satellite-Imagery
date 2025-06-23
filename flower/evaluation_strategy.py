
from collections import OrderedDict
import torch
from scripts.test import test_model
#from flower.flower_client import FlowerClient
from flwr.common import Metrics
from typing import List, Tuple
import os

best_psnr = 0.0
best_parameters = None
best_psnr_round = 0

def get_evalulate_fn(testloader, device, model, log_dir):
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""

    global best_psnr 
    global best_parameters
    global best_psnr_round
    
    def evaluate_fn(server_round: int, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        Then the model will be evaluated on the test set (recall this is the
        whole MNIST test set)."""

        global best_psnr 
        global best_parameters
        global best_psnr_round

        # model = RRDBNet(growth_channels = 2, num_rrdb= 1).to(device=device)
        #model = RRDBNet(upscale=2).to(device=device)
        # model = AsConvSR(device=device).to(device=device)
        
        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # call test
        
        psnr, ssim = test_model(model, testloader, device)
        
        if psnr > best_psnr:
            best_psnr = psnr
            best_parameters = parameters
            best_psnr_round = server_round
            # Save the best model parameters to disk
            print(f"New best psnr found in round {server_round}: {psnr}. Saving best model...")
            best_model_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
        else:
            print(f"No new best model found. Best PSNR till now {best_psnr} from round {best_psnr_round}.")
        
        return None, {"psnr": psnr, "ssim": ssim}

    return evaluate_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["psnr"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"psnr": sum(accuracies) / sum(examples)}