import numpy as np
import pandas as pd
import os, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import QPDataset, QPDatasetExtended
from model import MLPDenoiser, TransformerDenoiserPlus, MLPDiffusion, FF
#import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import argparse
import cvxpy as cp

def projectOntoFeasibleSet(X, A, b, ):
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
        A_i = A[i]
        b_i = b[i]
        x = X[i]
        z = cp.Variable(k)
        objective = cp.Minimize(cp.sum_squares(z - x))
        constraints = [A_i @ z <= b_i]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Projection failed at sample {i}: {prob.status}")

        Z_projected[i] = z.value

    return torch.from_numpy(Z_projected)


def batch_quadratic_objective(x, Q, p):
    # x: (n, k)
    # Q: (n, k, k)
    # c: (n, k)
    # x^T Q x = (x.unsqueeze(1) @ Q @ x.unsqueeze(2)).squeeze()
    quad_term = 0.5 * torch.einsum('ni,nij,nj->n', x, Q, x)  # shape: (n,)
    linear_term = torch.einsum('ni,ni->n', p, x)             # shape: (n,)
    print(x.requires_grad, x.grad_fn)
    return quad_term + linear_term                           # shape: (n,)


# Parse args

# 5 64 0 2 1 64 0.001 1 0.1 0.05 569
# 5 16 2 0 1 64 0.001 1 0.1 0.05 515

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
parser.add_argument("--use_guidance", type=bool, default = True, help="Add guidance term at sampling time")
parser.add_argument("--beta", type=float, default = 1e-2, help="(Initial) weight for guidance term")
parser.add_argument("--dynamic_beta", type=bool, default = False, help="If true, update beta during sampling")
parser.add_argument("--use_perSampleGradients", type=bool, default = False, help="If true, compute the objective gradient sample by sample individually; if false, averages across the batch")
parser.add_argument("--beta_step_size", type=float, default = 1e-3, help="Step size for beta")

parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
parser.add_argument("--hidden_units", type=int, default=64, help="Hidden units of the model")
parser.add_argument("--n_layers", type=int, default=5, help="Random seed")
parser.add_argument("--seed", type=int, default=2, help="Random seed")
parser.add_argument("--optimizer", type=int, default=2, help="Random seed")
parser.add_argument("--activation", type=int, default=1, help="Random seed")
parser.add_argument("--id", type=int, default=569, help="Hidden units of the model")
parser.add_argument("--id_script", type=int, default=-1, help="Hidden units of the model")

parser.add_argument("--sigma_min", type=float, default=0.05, help="Sigma min of Langevin dynamic")
parser.add_argument("--sigma_max", type=float, default=.1, help="Sigma max of Langevin dynamic")
parser.add_argument("--n_steps", type=int, default=2, help="Langevin steps")
parser.add_argument("--annealed_step", type=int, default=5, help="Annealed steps")

# Training parameters
parser.add_argument("--total_iteration", type=int, default=3000, help="Total training iterations")
parser.add_argument("--display_iteration", type=int, default=150, help="Logging frequency")
parser.add_argument("--run_name", type=str, default='train', help="Run name for logging and saving")
# Projection parameters

args = parser.parse_args()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

seed = args.seed
def set_seed(seed: int = 42):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  

# DEFINE DATALOADERS
folder = 'new_data'

train_data_path = f'{folder}/train_data_qp.csv'
if args.conditioning_type==2:
    dataset = QPDatasetExtended(train_data_path)
else:
    dataset = QPDataset(train_data_path)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

valid_data_path = f'{folder}/val_data_qp.csv'
if args.conditioning_type==2:
    dataset = QPDatasetExtended(valid_data_path)
else:
    dataset = QPDataset(valid_data_path)
valid_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

test_data_path = f'{folder}/test_data_qp.csv'
if args.conditioning_type==2:
    dataset = QPDatasetExtended(test_data_path)
else:
    dataset = QPDataset(test_data_path)
test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

A_perturbed = dataset.A_perturbed
b_perturbed = dataset.b_perturbed
Q_perturbed = dataset.Q_perturbed
p_perturbed = dataset.p_perturbed

A_unperturbed = torch.ones_like(A_perturbed) * -0.37039263882966883
Q_unperturbed = torch.ones_like(Q_perturbed) * 3.0
p_unperturbed = torch.ones_like(p_perturbed) * 1.0

tmp = torch.stack((A_unperturbed, A_perturbed), dim=1).squeeze()
A_numpy = tmp.numpy()
b_numpy = b_perturbed.squeeze()


n = Q_perturbed.shape[0]
Q = torch.zeros((n, 2, 2), device=Q_perturbed.device, dtype=Q_perturbed.dtype)
Q[:, 1, 1] = 3.0                          
Q[:, 0, 0] = Q_perturbed[:,0]                  

p = torch.zeros((p_perturbed.shape[0], 2), device=p_perturbed.device, dtype=p_perturbed.dtype)
p[:, 0] = 1.0                     # fixed component
p[:, 1] = p_perturbed[:,0]           # per-sample component

### DEFINE MODEL
for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)

n_output = x.shape[1]

if args.conditioning_type == 0:
    n_input = x.shape[1]
elif args.conditioning_type == 1:
    n_input = 1+y.shape[1]
else:
    n_input = x.shape[1]+1+y.shape[1]

layer_dims = [n_input] + [args.hidden_units] * args.n_layers + [n_output]

if args.activation == 0:
    act = nn.ReLU
elif args.activation == 1:
    act = nn.Tanh
elif args.activation == 2:
    act = nn.SiLU

print(y[:10,:])

model = FF([n_input,args.hidden_units,2], activation=act).to(device = device)
model.load_state_dict(torch.load(f'./models/{folder}/model_{args.id}.pt', map_location=torch.device('cpu')))

sigma_max = args.sigma_max
sigma_min = args.sigma_min
n_steps = args.n_steps
# sigmas = np.linspace(sigma_max, sigma_min, L, dtype=np.float32)
#sigmas = torch.exp
sigmas = torch.exp(torch.linspace(start=math.log(sigma_max), end=math.log(sigma_min), steps = n_steps)).to(device = device)
#print("Sigmas: ",sigmas)

eps = args.eps
sampling_list = []

#@torch.no_grad()
def sample_vectors(model, sigmas, n_steps, n_samples=1000, k=2, conditioning=None, conditioning_type=2, device='cuda'):
    """
    Generates samples using annealed Langevin dynamics from the trained score-based model.
    """
    model.eval()
    x = torch.randn((n_samples, k)).to(device)  # Initialize with random noise
    step_size = eps * (sigmas / sigmas[-1] ) ** 2
    #print("Step size: ",step_size)
    #step_size = step_size.flip(0)
    beta = args.beta

    for idx in range(len(sigmas)):   #
        #sigma = sigmas[idx]
        real_idx = len(sigmas)-idx-1
        for _ in range(args.annealed_step):  ### annealed_steps
            z = eps*torch.randn_like(x).to(device) ### eps in an hyperparams
            if conditioning_type == 0:
                x_in = x
            elif conditioning_type == 1:
                x_in = torch.cat([x, real_idx*torch.ones(x.shape[0]).unsqueeze(-1), y[:,2:]], dim=1)
            elif conditioning_type == 2:
                x_in = torch.cat([x, real_idx*torch.ones(x.shape[0]).unsqueeze(-1), y], dim=1)
            grad = model(x_in)

            if args.use_guidance:
                x.requires_grad_(True)
                obj = batch_quadratic_objective(x, Q, p)
                if args.use_perSampleGradients:
                    grads = []
                    for i in range(x.shape[0]):
                        grad_i = torch.autograd.grad(obj[i], x, retain_graph=True, create_graph=True)[0][i]
                        grads.append(grad_i)
                    obj_grad = torch.stack(grads, dim=0)  
                else:
                    obj_grad = torch.autograd.grad(obj.sum(), x, create_graph=True)[0]

                x = x + 0.5 * step_size[real_idx] * grad + torch.sqrt(step_size[real_idx]) * z  - beta * obj_grad # Langevin step
                if args.dynamic_beta:
                    beta = beta - args.beta_step_size *beta
            else:
                x = x + 0.5 * step_size[real_idx] * grad + torch.sqrt(step_size[real_idx]) * z  # Langevin step
            ### project into feasible set here
            if args.project:
                if args.use_guidance:
                    x = x.detach().numpy()
                x = projectOntoFeasibleSet(x,A_numpy,b_numpy)
            sampling_list.append(x)
    return sampling_list, x

samples, final_sample = sample_vectors(model, sigmas, args.annealed_step, n_samples=1000, k=2, device=device)
samples = torch.stack(samples).numpy()
np.save(f'allSamples/{folder}/samples_{args.id_script}.npy', np.array(samples))

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
df.to_csv(f'KL_{folder}.csv',mode='a', header=False, index=False)

