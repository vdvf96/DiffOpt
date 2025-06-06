import csv
import pandas as pd
# Replace this with your actual file path
file_path = 'MLP_DENOISER_TEST_LOSS_new_data_cpu.csv'

# Read and parse all rows
data = []
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            first_val = float(row[0])
            second_val = int(row[1])
            data.append((first_val, second_val))

# Sort by the first value
data.sort(key=lambda x: x[0])

# Print lowest 10
print("Lowest 150 entries:")
for val in data[:150]:
    print(f"Value: {val[0]:.6f}, Other: {val[1]}")

loss_df = pd.read_csv(file_path, header=None, names=["test loss", "id"])
loss_df["test loss"] = loss_df["test loss"].astype(float)
loss_df["id"] = loss_df["id"].astype(int)

'''
params_df = pd.read_csv("hyperparams_long_epochs.txt", header=None, sep="\s+")
# Filter rows where 4th value (index 3) is 2
filtered_params = params_df[params_df[3] == 2]
# Extract the last column as ID
valid_ids = filtered_params.iloc[:, -1]
# Filter the losses dataframe to only those IDs
filtered_losses = loss_df[loss_df["id"].isin(valid_ids)]
# Find the minimum loss and corresponding IDs
min_loss = filtered_losses["test loss"].min()
best_ids = filtered_losses[filtered_losses["test loss"] == min_loss]["id"].tolist()
'''

# Read losses.csv manually and split on comma
#loss_df = pd.read_csv("losses.csv", header=None)
#loss_df[[0, 1]] = loss_df[0].split(",") #, expand=True)
#loss_df.columns = ["loss", "id"]
#loss_df["loss"] = loss_df["loss"].astype(float)
#loss_df["id"] = loss_df["id"].astype(int)

# Read params.txt with space-separated values
params_df = pd.read_csv("hyperparams_long_epochs.txt", header=None, sep="\s+")
#hyperparams_long_epochs_cpu_1.txt

# Filter for rows where 4th value (index 3) is 2
filtered_params = params_df[params_df[3] == 1]

# Extract valid IDs from last column
valid_ids = filtered_params.iloc[:, -1].astype(int)

# Filter loss_df using valid_ids
filtered_losses = loss_df[loss_df["id"].isin(valid_ids)]

# Get min loss and corresponding ID(s)
min_loss = filtered_losses["test loss"].min()
best_ids = filtered_losses[filtered_losses["test loss"] == min_loss]["id"].tolist()

# Get the 10 rows with the smallest "test loss"
top_10 = filtered_losses.nsmallest(10, "test loss")

# Extract the corresponding IDs
best_ids = top_10["id"].tolist()

print("Minimum loss:", min_loss)
print("ID(s) with minimum loss (4th param = 2):", best_ids)
print("Minimum loss:", min_loss)
print("ID(s) with minimum loss (4th param = 2):", best_ids)
