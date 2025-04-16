import csv

# Replace this with your actual file path
file_path = 'MLP_DENOISER_TEST_LOSS.csv'

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
print("Lowest 10 entries:")
for val in data[:10]:
    print(f"Value: {val[0]:.6f}, Other: {val[1]}")