import numpy as np
import pandas as pd
import os, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.cm as cm
from dataloader import QPDataset
from model import MLPDenoiser, TransformerDenoiserPlus, MLPDiffusion, FF
#import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt

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
parser.add_argument("--activation", type=int, default=0, help="Random seed")
parser.add_argument("--id", type=int, default=367, help="Hidden units of the model")
parser.add_argument("--id_script", type=int, default=367, help="Hidden units of the model")
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

folder = 'easy_Data'
test_data_path = f'data/{folder}/test_data_qp.csv'
dataset = QPDataset(test_data_path)
test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

A_perturbed = dataset.A_perturbed
b_perturbed = dataset.b_perturbed

A_unperturbed = torch.ones_like(A_perturbed) * 0.5427325452178637
b_unperturbed = torch.ones_like(A_perturbed) *-0.7599301066980018

X1 = dataset.x[:,0]
X2 = dataset.x[:,1]

tmp = torch.stack((A_unperturbed, A_perturbed), dim=1).squeeze()
A_numpy = tmp.numpy()
tmp2 = (tmp * dataset.x).sum(dim=1)
tmp3 = tmp2 - b_perturbed.squeeze()

X_hat = np.load(f'samples/samples_{args.id_script}_longEpochs.npy')

X1_hat = X_hat[:,0]
X2_hat = X_hat[:,1]

tmp = torch.stack((A_unperturbed, A_perturbed), dim=1).squeeze()
tmp2 = (tmp * torch.from_numpy(X_hat)).sum(dim=1)
tmp3 = tmp2 - b_perturbed.squeeze()

# Mask for constraint satisfaction
satisfies_constraint = tmp3 < 0

# Violating points (X_hat and their violation magnitude)
X_viol = X_hat[~satisfies_constraint]
viol_amounts = tmp3[~satisfies_constraint].numpy()  # shape (num_viol,)
X1_viol = X_viol[:, 0]
X2_viol = X_viol[:, 1]

# Satisfying points
X_sat = X_hat[satisfies_constraint]
X1_sat = X_sat[:, 0]
X2_sat = X_sat[:, 1]


plt.figure(figsize=(8, 8))
# Violating samples — shaded by amount of violation
sc = plt.scatter(X1_viol, X2_viol, c=viol_amounts, cmap='Reds', s=10, alpha=0.6, marker='x', label='Generated (violate constraints)')
# Add colorbar for violations
cbar = plt.colorbar(sc)
cbar.set_label('Constraint Violation Amount')
# Satisfying samples
plt.scatter(X1_sat, X2_sat, s=10, alpha=0.3, color='green', marker='x', label='Generated (satisfy constraints)')
# Plot the "Original" points second — front layer
plt.scatter(X1, X2, s=20, alpha=0.8, color='blue', marker='o', label='Original', edgecolors='black')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
#plt.axis('equal')  # optional: keeps aspect ratio square
plt.savefig(f'plot/data/viol/longEpochs/viol_{args.id_script}.png')


# 0.5 * cp.quad_form(x, Q) + P.T @ x
Q = torch.tensor([[2,0],[0,3]]).float()
P = torch.tensor([1,1]).float()
X_hat_tensor = torch.from_numpy(X_hat)
#quad = X_hat_tensor @ Q @ X_hat_tensor.T  # scalar
tmp = torch.sum(X_hat_tensor,1)
hat_obj = .5*torch.einsum('bi,ij,bj->b', X_hat_tensor, Q, X_hat_tensor) + torch.sum(X_hat_tensor,1)
true_obj = .5*torch.einsum('bi,ij,bj->b', dataset.x, Q, dataset.x) + torch.sum(dataset.x,1)

gap = 100 * torch.abs(true_obj - hat_obj) / torch.abs(true_obj)
gap = gap.numpy()  # convert to numpy for plotting

print(gap)
print(np.mean(gap))
print(np.max(gap))
log_gap = np.log10(gap + 1e-6)

plt.figure(figsize=(8, 6))
sc = plt.scatter(X1_hat, X2_hat, c=log_gap, cmap='plasma', s=15, alpha=0.8, marker='x')

# Colorbar with custom ticks showing % values
cbar = plt.colorbar(sc)
cbar.set_label('log10(Optimality Gap %)')
cbar.set_ticks([0, 1, 2, 3, 4, 5])  # Optional: customize to match your range
cbar.set_ticklabels([r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Samples Colored by log Optimality Gap')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'plot/data/gap/longEpochs/log_gap_{args.id_script}.png', dpi=300)
plt.show()

'''
x_vals = torch.linspace(-5, 5, 200)

b = b_perturbed
print(b.min())
print(b.max())
print(A_numpy.min())
print(A_numpy.max())

# Normalize b values for colormap mapping
b_norm = (b - b.min()) / (b.max() - b.min())
colors = cm.viridis(b_norm.numpy())  # or try 'plasma', 'cividis', 'magma'

plt.figure(figsize=(8, 6))

for i in range(A_numpy.shape[0]):
    a1, a2 = A_numpy[i]
    b_i = b[i]
    color = colors[i]

    if a2 != 0:
        y_vals = (b_i - a1 * x_vals) / a2
        plt.plot(x_vals, y_vals, color=color, alpha=0.4)
    else:
        x_const = b_i / a1
        plt.axvline(x_const, color=color, alpha=0.4)

plt.title("Constraint Lines $Ax = b$ (Shaded by $b$)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plot/data/constraint_lines_shaded_{args.id_script}.png", dpi=300)
plt.show()
'''