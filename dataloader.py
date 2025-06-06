import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset



def parse_array(s):
    return np.array(eval(s), dtype=np.float32)


class QPDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)

        x = np.stack(df['x'].apply(parse_array))

        A_perturbed = df["A[0,1]"].values.astype(np.float32).reshape(-1, 1)
        b_perturbed = df["b[0]"].values.astype(np.float32).reshape(-1, 1)

        self.x = torch.tensor(x, dtype=torch.float32)
        self.A_perturbed = torch.tensor(A_perturbed, dtype=torch.float32)
        self.b_perturbed = torch.tensor(b_perturbed, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        target = self.x[idx]
        condition = torch.cat([self.A_perturbed[idx], self.b_perturbed[idx]])
        return target, condition
    


class QPDatasetExtended(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)

        x = np.stack(df['x'].apply(parse_array))

        A_perturbed = df["A[0,1]"].values.astype(np.float32).reshape(-1, 1)
        b_perturbed = df["b[0]"].values.astype(np.float32).reshape(-1, 1)
        Q_perturbed = df["Q[0,0]"].values.astype(np.float32).reshape(-1, 1)
        p_perturbed = df["p[1]"].values.astype(np.float32).reshape(-1, 1)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.A_perturbed = torch.tensor(A_perturbed, dtype=torch.float32)
        self.b_perturbed = torch.tensor(b_perturbed, dtype=torch.float32)
        self.Q_perturbed = torch.tensor(Q_perturbed, dtype=torch.float32)
        self.p_perturbed = torch.tensor(p_perturbed, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        target = self.x[idx]
        condition = torch.cat([self.A_perturbed[idx], self.b_perturbed[idx], self.Q_perturbed[idx], self.p_perturbed[idx]])
        return target, condition