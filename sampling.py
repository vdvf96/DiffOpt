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
import cvxpy as cp

def projectOntoFeasibleSet(X, A, b):
    """
    Projects a batch of points X (n x k) into the feasible set defined by Ax <= b
    for each row individually.

    Args:
        X  Array of shape (n, k), n points of dimension k
        A  Constraint matrix of shape (k)
        b  Constraint vector of shape (1)

    Returns:
        Z_projected Projected points of shape (n, k)
    """
    n, k = X.shape
    Z_projected = np.zeros_like(X)

    for i in range(n):
        x = X[i]
        z = cp.Variable(k)
        objective = cp.Minimize(cp.sum_squares(z - x))
        constraints = [A @ z <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Projection failed at sample {i}: {prob.status}")

        Z_projected[i] = z.value

    return Z_projected


def objective(x, Q, p):
    return .5* x @ Q @ x + p @ x


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
parser.add_argument("--project", type=bool, default = True, help="Enable projection step at sampling time")
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
parser.add_argument("--annealed_step", type=int, default=25, help="Annealed steps")

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

A_perturbed = dataset.A_perturbed
b_perturbed = dataset.b_perturbed
#Q_perturbed = dataset.

A_unperturbed = torch.ones_like(A_perturbed) * 0.5427325452178637

tmp = torch.stack((A_unperturbed, A_perturbed), dim=1).squeeze()
A_numpy = tmp.numpy()
b_numpy = b_perturbed.squeeze()

n_output = 2

if args.conditioning_type == 0:
    n_input = 2
elif args.conditioning_type == 1:
    n_input = 3
else:
    n_input = 5

layer_dims = [n_input] + [args.hidden_units] * args.n_layers + [n_output]

if args.activation == 0:
    act = nn.ReLU
elif args.activation == 1:
    act = nn.Tanh
elif args.activation == 2:
    act = nn.SiLU


### DEFINE MODEL
for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)

folder = 'easy_Data'
model = FF([n_input,args.hidden_units,2], activation=act).to(device = device)
model.load_state_dict(torch.load(f'./models/{folder}/model_{args.id}.pt'))

sigma_max = 0.005
sigma_min = 10
n_steps = args.n_steps
# sigmas = np.linspace(sigma_max, sigma_min, L, dtype=np.float32)
sigmas = torch.exp(torch.linspace(start=math.log(sigma_max), end=math.log(sigma_min), steps = n_steps)).to(device = device)
print(sigmas)

eps = args.eps
sampling_list = []

@torch.no_grad()
def sample_vectors(model, sigmas, n_steps, n_samples=1000, k=2, conditioning=None, conditioning_type=2, device='cuda'):
    """
    Generates samples using annealed Langevin dynamics from the trained score-based model.
    """
    model.eval()
    x = torch.randn((n_samples, k)).to(device)  # Initialize with random noise
    step_size = (eps * (sigmas / sigmas[-1] ) ** 2)
    step_size = step_size.flip(0)
    print(step_size)
    for idx in range(len(sigmas)):   #
        sigma = sigmas[idx]

        for _ in range(args.annealed_step):  ### annealed_steps
            z = eps*torch.randn_like(x).to(device) ### eps in an hyperparams
            if conditioning_type == 0:
                x_in = x
            elif conditioning_type == 1:
                x_in = torch.cat([x, idx*torch.ones(x.shape[0]).unsqueeze(-1)], dim=1)
            elif conditioning_type == 2:
                x_in = torch.cat([x, idx*torch.ones(x.shape[0]).unsqueeze(-1), y], dim=1)
            grad = model(x_in)
            x.requires_grad_(True)
            x = x + 0.5 * step_size[idx] * grad + torch.sqrt(step_size[idx]) * z  # Langevin step

            ### project into feasible set here
            if args.project:
                x = projectOntoFeasibleSet(x, A_numpy, b_numpy)
            sampling_list.append(x)
    return sampling_list, x

samples, final_sample = sample_vectors(model, sigmas, args.annealed_step, n_samples=1000, k=2, device=device)
np.save(f'samples/samples_{args.id_script}_longEpochs.npy', final_sample.detach().numpy())

'''
record = {
    'test loss' : [test_loss],
    'id' : [args.id]   
}

df = pd.DataFrame(record)
df.to_csv('MLP_DENOISER_TEST_LOSS_LONG_EPOCHS_DIFFICULT_DATA.csv',mode='a', header=False, index=False)

'''