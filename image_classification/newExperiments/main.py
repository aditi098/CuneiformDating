import torch
import torch.nn as nn
import torch.optim as optim
from experiment import Experiment
import json


if __name__ == "__main__":
    exp_name = "default"
    #     if len(sys.argv) > 1:
    #         exp_name = sys.argv[1]
    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    exp.train()
    exp.test()