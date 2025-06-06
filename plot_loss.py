import numpy as np
import matplotlib.pyplot as plt
import argparse

# Load saved arrays
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="Run name for logging and saving")
# Projection parameters

args = parser.parse_args()
#device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
folder = 'new_data'
id = args.id
train_loss = np.load(f"loss/{folder}/train_loss_list_{id}.npy")  # shape: (n_epochs,)
val_loss = np.load(f"loss/{folder}/valid_loss_list_{id}.npy")      # shape: (n_epochs,)

if id==299 or id==434 or id==407 or id==119 or id==182 or id==443 or id==83 or id==290 or id==326 or id==191:
    title = "Constraint-unaware model"
else:
    title = "Objective-aware model"
# Plot
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss, {title}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'plot/loss/{folder}/losses_{id}.png')
#plt.show()
