import numpy as np
import pandas as pd
import os, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import QPDataset
from model import MLPDenoiser, TransformerDenoiserPlus, MLPDiffusion, FF
#import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import argparse


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=['train', 'sample'], default='train', help="Training or sampling") #
# Setting parameters
parser.add_argument("--experiment", type=str, choices=['microstructures', 'trajectories', 'motion', 'human', 'other','QP'], default='microstructures', help="Experiment setting")
parser.add_argument("--projection_path", type=str, default=None, help="If experiment argument is \'other\', set path to custom projection operator")
parser.add_argument("--model_path", type=str, default=None, help="Set path to diffusion model checkpoint")
parser.add_argument("--train_set_path", type=str, default=None, help="Set path to training data if in training mode")
parser.add_argument("--val_set_path", type=str, default=None, help="Set path to validation data if in training mode")
parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
parser.add_argument("--batch_size", type=int, default=128, help="Random seed")
parser.add_argument("--max_epochs", type=int, default=100000, help="Random seed")
parser.add_argument("--conditioning_type", type=int, default=2, help="Random seed")
# Model parameters
parser.add_argument("--eps", type=float, default=1.5e-5, help="Epsilon of step size")

parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
parser.add_argument("--hidden_units", type=int, default=64, help="Hidden units of the model")
parser.add_argument("--n_layers", type=int, default=1, help="Random seed")
parser.add_argument("--seed", type=int, default=2, help="Random seed")
parser.add_argument("--optimizer", type=int, default=2, help="Random seed")
parser.add_argument("--activation", type=int, default=1, help="Random seed")
parser.add_argument("--id", type=int, default=367, help="Hidden units of the model")
parser.add_argument("--id_script", type=int, default=-1, help="Hidden units of the model")

parser.add_argument("--sigma_min", type=float, default=0.005, help="Sigma min of Langevin dynamic")
parser.add_argument("--sigma_max", type=float, default=10., help="Sigma max of Langevin dynamic")
parser.add_argument("--n_steps", type=int, default=10, help="Langevin steps")
parser.add_argument("--annealed_step", type=int, default=20, help="Annealed steps")

# Training parameters
parser.add_argument("--total_iteration", type=int, default=3000, help="Total training iterations")
parser.add_argument("--display_iteration", type=int, default=150, help="Logging frequency")
parser.add_argument("--run_name", type=str, default='train', help="Run name for logging and saving")
# Projection parameters

args = parser.parse_args()
device = torch.device('cpu') #torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

seed = args.seed
def set_seed(seed: int = 42):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  

# DEFINE DATALOADERS
folder = 'easy_Data'

train_data_path = f'data/{folder}/train_data_qp.csv'
dataset = QPDataset(train_data_path)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

valid_data_path = f'data/{folder}/val_data_qp.csv'
dataset = QPDataset(valid_data_path)
valid_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

test_data_path = f'data/{folder}/test_data_qp.csv'
dataset = QPDataset(test_data_path)
test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

### DEFINE MODEL
for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)

final_sample = np.load(f'allSamples/{folder}/samples_{args.id_script}.npy')[-1,:,:]
print("DONE")


def estimate_2d_histogram(samples, bins=50, range=None):
    """
    Turns (n,2) sample points into a normalized 2D histogram (discrete distribution).
    """
    hist, xedges, yedges = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=bins,
        range=range,
        density=False
    )
    hist = hist.astype(np.float32)
    hist += 1e-10  # avoid zero entries
    hist /= hist.sum()  # normalize to make it a probability distribution
    return torch.tensor(hist), xedges, yedges
    
def kl_between_histograms(p_hist: torch.Tensor, q_hist: torch.Tensor):
    p = p_hist / p_hist.sum()
    q = q_hist / q_hist.sum()
    kl = p * (torch.log(p) - torch.log(q))
    return kl.sum()


combined = np.vstack([x, final_sample])
xy_min, xy_max = combined.min(axis=0), combined.max(axis=0)
hist_range = [[xy_min[0], xy_max[0]], [xy_min[1], xy_max[1]]]

# estimate histograms
p_hist, _, _ = estimate_2d_histogram(x, bins=50, range=hist_range)
q_hist, _, _ = estimate_2d_histogram(final_sample, bins=50, range=hist_range)

# compute KL divergence
kl = kl_between_histograms(p_hist, q_hist)

record = {
    'KL' : [kl.item()],
    'id' : [args.id_script]   
}

df = pd.DataFrame(record)
df.to_csv('KL.csv',mode='a', header=False, index=False)