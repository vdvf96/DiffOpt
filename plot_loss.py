import numpy as np
import matplotlib.pyplot as plt
import argparse

# Load saved arrays
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="Run name for logging and saving")
# Projection parameters

args = parser.parse_args()
#device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
id = args.id
train_loss = np.load(f"loss/train_loss_list_{id}.npy")  # shape: (n_epochs,)
val_loss = np.load(f"loss/valid_loss_list_{id}.npy")      # shape: (n_epochs,)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'plot/losses_{id}.png')
#plt.show()
