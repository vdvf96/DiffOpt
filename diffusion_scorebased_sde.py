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
parser.add_argument("--max_epochs", type=int, default=300000, help="Random seed")
parser.add_argument("--conditioning_type", type=int, default=2, help="Random seed")
# Model parameters
parser.add_argument("--eps", type=float, default=1.5e-5, help="Epsilon of step size")

parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
parser.add_argument("--hidden_units", type=int, default=20, help="Hidden units of the model")
parser.add_argument("--n_layers", type=int, default=2, help="Random seed")
parser.add_argument("--seed", type=int, default=2, help="Random seed")
parser.add_argument("--optimizer", type=int, default=2, help="Random seed")
parser.add_argument("--activation", type=int, default=0, help="Random seed")
parser.add_argument("--id", type=int, default=20, help="Hidden units of the model")

parser.add_argument("--sigma_min", type=float, default=0.005, help="Sigma min of Langevin dynamic")
parser.add_argument("--sigma_max", type=float, default=10., help="Sigma max of Langevin dynamic")
parser.add_argument("--n_steps", type=int, default=10, help="Langevin steps")
parser.add_argument("--annealed_step", type=int, default=25, help="Annealed steps")
parser.add_argument("--normalize", type=int, default=0, help="Data normalization")

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

folder = 'difficult_Data'

train_data_path = f'data/{folder}/train_data_qp.csv'
dataset = QPDataset(train_data_path)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

valid_data_path = f'data/{folder}/val_data_qp.csv'
dataset = QPDataset(valid_data_path)
valid_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)


test_data_path = f'data/{folder}/test_data_qp.csv'
dataset = QPDataset(test_data_path)
test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)


n_output = 2
is_y_cond=True

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


model = FF([n_input,args.hidden_units,2], activation=act).to(device = device)

print(model)

if args.optimizer == 0:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 1:
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 2:
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
#optimizer = optim.AdamW(model.parameters(), lr=args.lr)


epochs = args.max_epochs
patience = 0
max_patience = 1000

min_loss = float('inf')
sigma_max = 0.005
sigma_min = 10
n_steps = 10
# sigmas = np.linspace(sigma_max, sigma_min, L, dtype=np.float32)
sigmas = torch.exp(torch.linspace(start=math.log(sigma_max), end=math.log(sigma_min), steps = n_steps)).to(device = device)
epoch_loss = []
valid_loss_list = []
test_loss = 0

###
### HYPERPARAMS
###
########################
###
### NUMBER OF LAYERS, UNITS - DONE
### ACTIVATION FUNCTION - DONE
### DROPOUT - LET'S SEE IF NEEDED
### LEARNING RATE - DONE
### BATCH SIZE - DONE
### SIGMAS (NOISE LEVEL) - LET'S SEE IF NEEDED
### EARLY STOPPING - DONE 
### LOSS FUNCTION (WEIGHTED BY NOISE LEVEL OR NOT) - LET'S SEE IF NEEDED
### OPTIMIZER (ADAM, SGD, RMSPROP) - DONE
### POSSIBLY DATA GEN - LET'S SEE IF NEEDED
### DATA NORMALIZATION - TO DO
### VARIOUS CONDITIONING - DONE
###
### PROBLEM STRUCTURE
###

# min y = 1/2 xQx + px 
# Ax =b
# x | A, b

def min_max_normalize(t):
    return (t - t.min(dim=0, keepdim=True).values) / (t.max(dim=0, keepdim=True).values - t.min(dim=0, keepdim=True).values + 1e-8)

for epoch in range(epochs):
    running_loss = 0.0
    print('='* 80)
    iteration = 0 
    model.train()
    for x, y in dataloader:
        if args.normalize==1:
            x = min_max_normalize(x)
            y = min_max_normalize(y)
        x, y = x.to(device), y.to(device)
        # print(x)
        # print(y)
        idx = torch.randint(0, len(sigmas), (x.size(0), 1)).to(device = device)
        #idx = torch.randint(0, len(sigmas)).to(device = device)
        chosen_sigmas = sigmas[idx] # shape (batch, 1)
        noise = torch.randn_like(x, device=device)
        x_tilde = x + chosen_sigmas * noise
        #target_score = (x - x_tilde) / (chosen_sigmas**2)
        target_score = -1/(chosen_sigmas) * noise # same thing as above expression 
        
        #pred_score = model(x_tilde, idx , y) ### NOISE AND PROBLEM PARAEMTERS CONDITIONING
        #pred_score = model(x_tilde, idx)  ### NOISE CONDITIONING

        if args.conditioning_type == 0:
            x_in = x_tilde
        elif args.conditioning_type == 1:
            x_in = torch.cat([x_tilde, idx], dim = 1)
        elif args.conditioning_type == 2:
            x_in = torch.cat([x_tilde, idx, y], dim = 1)
        
        pred_score = model(x_in) ### NO CONDITIONING
        
        # loss = (pred_score - target_score).square().mean()
        loss = (torch.square(target_score - pred_score).mean(-1) * chosen_sigmas**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
        iteration+=1
        # if iteration %5 == 0: 
        #     print('Checkpoint: Batch loss:{}  Running_loss: {}'.format(loss.item(), running_loss/iteration))
    
    avg_loss = running_loss / len(dataset)
    epoch_loss.append(avg_loss)
    if epoch % 1 ==0: 
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        model.eval()
        with torch.no_grad():
            for x, y in valid_dataloader:
                x, y = x.to(device), y.to(device)
                if args.normalize==1:
                    x = min_max_normalize(x)
                    y = min_max_normalize(y)
                # print(x)
                # print(y)
                idx = torch.randint(0, len(sigmas), (x.size(0), 1)).to(device = device)
                #idx = torch.randint(0, len(sigmas)).to(device = device)
                chosen_sigmas = sigmas[idx] # shape (batch, 1)
                noise = torch.randn_like(x, device=device)
                x_tilde = x + chosen_sigmas * noise
                #target_score = (x - x_tilde) / (chosen_sigmas**2)
                target_score = -1/(chosen_sigmas) * noise # same thing as above expression 
                
                #pred_score = model(x_tilde, idx , y) ### NOISE AND PROBLEM PARAEMTERS CONDITIONING
                #pred_score = model(x_tilde, idx)  ### NOISE CONDITIONING

                if args.conditioning_type == 0:
                    x_in = x_tilde
                elif args.conditioning_type == 1:
                    x_in = torch.cat([x_tilde, idx], dim = 1)
                elif args.conditioning_type == 2:
                    x_in = torch.cat([x_tilde, idx, y], dim = 1)
                
                pred_score = model(x_in) ### NO CONDITIONING
                # loss = (pred_score - target_score).square().mean()
                loss = (torch.square(target_score - pred_score).mean(-1) * chosen_sigmas**2).mean()
                valid_loss = loss.item()

            if valid_loss<min_loss:
                min_loss = valid_loss
                print('Saving model...')
                torch.save(model.state_dict(), f'./models/{folder}/model_{args.id}_norm.pt')
                print('Model saved.')
                patience = 0
            else:
                patience += 1
            if patience == max:
                print('Early stopping...')
                break
            valid_loss_list.append(valid_loss)

np.save(f'loss/{folder}/valid_loss_list_{args.id}_norm.npy', valid_loss_list)  
np.save(f'loss/{folder}/train_loss_list_{args.id}_norm.npy', epoch_loss)

model.load_state_dict(torch.load(f'./models/{folder}/model_{args.id}_norm.pt'))
model.eval()

for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    if args.normalize==1:
        x = min_max_normalize(x)
        y = min_max_normalize(y)
    # print(x)
    # print(y)
    idx = torch.randint(0, len(sigmas), (x.size(0), 1)).to(device = device)
    #idx = torch.randint(0, len(sigmas)).to(device = device)
    chosen_sigmas = sigmas[idx] # shape (batch, 1)
    noise = torch.randn_like(x, device=device)
    x_tilde = x + chosen_sigmas * noise
    #target_score = (x - x_tilde) / (chosen_sigmas**2)
    target_score = -1/(chosen_sigmas) * noise # same thing as above expression 
    
    #pred_score = model(x_tilde, idx , y) ### NOISE AND PROBLEM PARAEMTERS CONDITIONING
    #pred_score = model(x_tilde, idx)  ### NOISE CONDITIONING

    if args.conditioning_type == 0:
        x_in = x_tilde
    elif args.conditioning_type == 1:
        x_in = torch.cat([x_tilde, idx], dim = 1)
    elif args.conditioning_type == 2:
        x_in = torch.cat([x_tilde, idx, y], dim = 1)
    
    pred_score = model(x_in) ### NO CONDITIONING
    # loss = (pred_score - target_score).square().mean()
    loss = (torch.square(target_score - pred_score).mean(-1) * chosen_sigmas**2).mean()
    test_loss = loss.item()

record = {
    'test loss' : [test_loss],
    'id' : [args.id]   
}

df = pd.DataFrame(record)
df.to_csv('MLP_DENOISER_TEST_LOSS_LONG_EPOCHS_AND NORMALIZE.csv',mode='a', header=False, index=False)

# df_res = pd.DataFrame({'train_epoch_loss':epoch_loss})
# df_res.to_csv('diffusion_mlpadvanced_100k.csv', index=False)
# plt.plot(x=range(len(epoch_loss)), y=epoch_loss)
# plt.show()

# val_r, val_alpha, val_beta, val_x = r[1500:], alpha[1500:], beta[1500:], x[1500:]

# datase_val = PortfolioDataset(val_r, val_alpha, val_beta, val_x)
# dataloader = DataLoader(datase_val, batch_size=16, shuffle=True)

# batch_opt, batch_cond = next(iter(dataloader))

# gen = annealed_langevin_sampling(unet, batch_cond[:2],score_scheduler, num_samples=2)
# print("Generated shape:", gen.shape)  # [2, 1, 512]
# print('Check:', F.mse_loss(batch_opt[0], gen))
