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
import random
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import imageio

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
parser.add_argument("--id_script", type=int, default=120, help="Hidden units of the model")
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

folder = 'new_data'
test_data_path = f'{folder}/test_data_qp.csv'
dataset = QPDataset(test_data_path)
test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# Pre-stack A and b once
A_perturbed   = dataset.A_perturbed[:100]                # (batch, 2)
A_unperturbed = torch.ones_like(A_perturbed) * 1.2100779822701702 #0.5427325452178637
b_perturbed   = dataset.b_perturbed.squeeze()[:100]       # (batch,)
A_stack       = torch.stack((A_unperturbed, A_perturbed), dim=1).squeeze()[:100]  # (batch, 2)
Q = [[0,31173912564542305],[0,3]]
P = [1,0.08736150635077833]

#Q,P,A_base,b_base
#"[[0.31173912564542305, 0.0], [0.0, 3.0]]","[1.0, 0.08736150635077833]","[[-0.37039263882966883, 1.2100779822701702]]",[-0.6277599050453051]

#Q,P,A_base,b_base
#[[-0.37039263882966883, 1.2100779822701702]]",[-0.6277599050453051]
# Q[0,0], p[1], A[0,1], b

def make_gif_sample_with_constraints(
    X_hat: np.ndarray,
    sample_idx: int,
    save_path: str,
    fps: int = 10
):
    """
    X_hat: np array of shape (T, batch, 2)
    sample_idx: which sample in the batch to track
    """
    frames = []
    # dynamic axis limits per sample
    x_min, x_max = X_hat[:,sample_idx,0].min() - 1, X_hat[:,sample_idx,0].max() + 1
    y_min, y_max = X_hat[:,sample_idx,1].min() - 1, X_hat[:,sample_idx,1].max() + 1

    # precompute objective contour grid
    Xg, Yg = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    Z = 0.5 * (Q[0][0]*Xg**2 + 2*Q[0][1]*Xg*Yg + Q[1][1]*Yg**2) + P[0]*Xg + P[1]*Yg

    # extract this sample's A and b
    a_vec = A_stack[sample_idx].numpy()
    b_val = b_perturbed[sample_idx].item()
    x_vals = np.array([x_min, x_max], dtype=float)
    if abs(a_vec[1]) > 1e-8:
        y_vals = (b_val - a_vec[0]*x_vals) / a_vec[1]
    else:
        x_vals = np.array([b_val/a_vec[0]]*2)
        y_vals = np.array([y_min, y_max])

    for t in range(X_hat.shape[0]):
        x_t = torch.from_numpy(X_hat[t, sample_idx]).float()
        val = (A_stack[sample_idx] * x_t).sum() - b_perturbed[sample_idx]

        fig, ax = plt.subplots(figsize=(5,5))
        # contour plot of objective
        cs = ax.contour(Xg, Yg, Z, levels=10, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8)

        # boundary line
        ax.plot(x_vals, y_vals, '--', linewidth=2, label='Boundary')

        # sample marker
        if val >= 0:
            sc = ax.scatter(
                x_t[0].item(), x_t[1].item(),
                c=[val.item()], cmap='Reds',
                s=50, marker='x', label='Violation'
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Violation Amount')
        else:
            ax.scatter(
                x_t[0].item(), x_t[1].item(),
                color='green', s=50, marker='x', label='Feasible'
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Sample {sample_idx}: Step {t+1}/{X_hat.shape[0]}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right')
        ax.grid(True)

        # capture frame
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        h, w = fig.canvas.get_width_height()
        frames.append(img.reshape((h, w, 3)))
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=fps)
    print(f"→ Single-sample GIF saved to {save_path}")


def make_gif_batch_with_constraints(
    X_hat: np.ndarray,
    save_path: str,
    fps: int = 10
):
    """
    X_hat: np array of shape (T, batch, 2)
    """
    frames = []
    # optionally truncate batch
    X_hat = X_hat[:, :100, :]

    # placeholder axis limits; will update per frame
    # but use for contour grid initial extent
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5

    # precompute objective contour grid
    Xg, Yg = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    Z = 0.5 * (Q[0][0]*Xg**2 + 2*Q[0][1]*Xg*Yg + Q[1][1]*Yg**2) + P[0]*Xg + P[1]*Yg

    for t in range(X_hat.shape[0]):
        X_t = torch.from_numpy(X_hat[t]).float()
        # dynamic axes
        xi_min, xi_max = X_t[:,0].min()-1, X_t[:,0].max()+1
        yi_min, yi_max = X_t[:,1].min()-1, X_t[:,1].max()+1

        vals = (A_stack * X_t).sum(dim=1) - b_perturbed
        violates = vals >= 0
        sats = vals < 0

        fig, ax = plt.subplots(figsize=(5,5))
        # contour
        cs = ax.contour(Xg, Yg, Z, levels=10, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8)

        # boundaries per sample
        for i in range(A_stack.shape[0]):
            a_vec = A_stack[i].numpy()
            b_val = b_perturbed[i].item()
            xb = np.array([xi_min, xi_max], dtype=float)
            if abs(a_vec[1]) > 1e-8:
                yb = (b_val - a_vec[0]*xb) / a_vec[1]
            else:
                xb = np.array([b_val/a_vec[0]]*2)
                yb = np.array([yi_min, yi_max])
            ax.plot(xb, yb, '--', color='gray', alpha=0.3)

        # scatter violators and feasibles
        if violates.any():
            sc = ax.scatter(
                X_t[violates,0].numpy(), X_t[violates,1].numpy(),
                c=vals[violates].numpy(), cmap='Reds',
                s=10, alpha=0.6, marker='x', label='Violating'
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Violation Amount')
        if sats.any():
            ax.scatter(
                X_t[sats,0].numpy(), X_t[sats,1].numpy(),
                color='green', marker='x', s=5, alpha=0.3,
                label='Satisfying'
            )

        ax.set_xlim(xi_min, xi_max)
        ax.set_ylim(yi_min, yi_max)
        ax.set_title(f"Batch: Step {t+1}/{X_hat.shape[0]}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right')
        ax.grid(True)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        h, w = fig.canvas.get_width_height()
        frames.append(img.reshape((h, w, 3)))
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=fps)
    print(f"→ Batch GIF saved to {save_path}")


X_hat = np.load(f'allSamples/{folder}/samples_{args.id_script}.npy')

# 1) Single sample (e.g. sample index 0)
make_gif_sample_with_constraints(
    X_hat=X_hat,
    sample_idx=0,
    save_path=f'plot/data/{folder}/allSamples/evolution_sample0_{args.id_script}.gif',
    fps=8
)

# 2) Full batch
make_gif_batch_with_constraints(
    X_hat=X_hat,
    save_path=f'plot/data/{folder}/allSamples/evolution_batch_{args.id_script}.gif',
    fps=16
)