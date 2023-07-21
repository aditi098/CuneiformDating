import torch
import torch.nn as nn
import torch.optim as optim
from experiment import Experiment
import json
import sys

if __name__ == "__main__":
    exp_name = "default"
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    
    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
#     exp.load_experiment("results/2023-07-07 02:22:49.906261/default_with_balanced_data.pt")
    exp.train()
    exp.test()
    exp.test(val=True)