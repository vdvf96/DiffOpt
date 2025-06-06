import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cvxpy as cp



def generate_qp_samples(Q, P, n=2, m=1, num_samples=10000):
    """
    Generate synthetic QP data with fixed Q, P, and y, varying (x*, A, b),
    ensuring a fixed optimal objective value for multiple instances with high precision.

    Parameters:
        Q (np.ndarray): Fixed n x n negative definite matrix.
        P (np.ndarray): Fixed n-dimensional vector.
        n (int): Number of decision variables.
        m (int): Number of equality constraints.
        num_samples (int): Number of samples to be run.
        tol (float): Tolerance for ensuring computed objective value matches the fixed value.

    Returns:
        DataFrame containing (optimal_obj, x_opt, A, b, computed_obj) for all samples.
    """
    results = []
    variables = []

    A_base = np.random.randn(m, n)
    b_base = np.random.randn(m)
    Q_base = Q
    P_base = P

    while len(results) < num_samples:
        Q[0,0] = np.random.uniform(0,1)
        P[1] = np.random.uniform(0,1)
        A = A_base.copy()
        b = b_base.copy()
        A[0, 1] *= np.random.uniform(1, 2)
        b[0] *= np.random.uniform(1, 2)
        x = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + P.T @ x)

        constraints = [A @ x <= b]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        if x.value is None:
            continue

        x_opt = x.value
        #computed_obj = 0.5 * x_opt.T @ Q @ x_opt + P.T @ x_opt
        
        results.append((x_opt.tolist(), A[0, 1], b[0], Q[0,0], P[1]))

    df_results = pd.DataFrame(
        results, columns=["x", "A_perturbed_01", "b_perturbed_0", "Q_perturbed_00", "P_perturbed_1"]
    )

    return df_results, A_base, b_base, Q_base, P_base


##################################################
# generate data
##################################################
# Q = np.random.randn(2,2)

# Q = -Q.T @ Q
 
Q_base = np.array([[2.0, 0], [0, 3.0]])
P_base = np.array([1.0, 1.0]) # np.random.rand(2)
df, A_base, b_base, _, _ = generate_qp_samples( Q_base, P_base, n=2, m=1, num_samples=10000)

##################################################
#save data
##################################################

x =  np.vstack(df['x'].values)
#opt_val = 0.5 * np.einsum("ij, jk, ik->i", x, Q, x) + np.einsum('j, ij->i', P, x)
# Randomly choose test indices
all_idx = np.arange(10000)

test_idx = np.random.choice(all_idx, 1000, replace=False)

# Remaining indices after test
remaining_idx = np.setdiff1d(all_idx, test_idx)

# Randomly choose validation indices from remaining
val_idx = np.random.choice(remaining_idx, 1000, replace=False)

# Remaining are training indices
train_idx = np.setdiff1d(remaining_idx, val_idx)

# Final counts
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
samples = []
os.makedirs('./new_data', exist_ok=True)


data_list = []
for idx in train_idx:
    x, A, b, q, p = df.iloc[idx].values
    data_list.append({
        "A[0,1]": A,
        "b[0]": b,
        "Q[0,0]": q,
        "p[1]": p,
        "x": x,
    })
df_save = pd.DataFrame(data_list)
df_save.to_csv('./new_data/train_data_qp.csv', index=False)


val_data_list = []
for idx in val_idx:
    x, A, b, q, p = df.iloc[idx].values
    val_data_list.append({
        "A[0,1]": A,
        "b[0]": b,
        "Q[0,0]": q,
        "p[1]": p,
        "x": x,
    })
df_val = pd.DataFrame(val_data_list)
df_val.to_csv('./new_data/val_data_qp.csv', index=False)


test_data_list = []
for idx in test_idx:
    x, A, b, q, p = df.iloc[idx].values
    test_data_list.append({
        "A[0,1]": A,
        "b[0]": b,
        "Q[0,0]": q,
        "p[1]": p,
        "x": x,
    })
df_test = pd.DataFrame(test_data_list)
df_test.to_csv('./new_data/test_data_qp.csv', index=False)


df_test_vars = pd.DataFrame({
    "Q": [Q_base.tolist()],
    "P": [P_base.tolist()],
    "A_base": [A_base.tolist()],
    "b_base": [b_base.tolist()]
})
df_test_vars.to_csv('./new_data/qp_base_variables.csv', index=False)

##################################################
# plot contour
##################################################

x = np.array(df['x'].tolist())
X1 = x[:, 0]
X2 = x[:, 1]

# Extract perturbed A[0,1] values
A_perturbed_vals = df["A_perturbed_01"].values
b_vals = df["b_perturbed_0"].values

Y1 = np.zeros_like(X1)          # Cluster 1 at y = 0
Y2 = np.ones_like(X2) * 1.0     # Cluster 2 at y = 1
'''
# Plot
plt.figure(figsize=(8, 2))
plt.scatter(X1, Y1, label='Cluster 1', alpha=0.5, s=10, color='blue')
plt.scatter(X2, Y2, label='Cluster 2', alpha=0.5, s=10, color='orange')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Two Clusters in 2D')
plt.legend()
plt.grid(True)
plt.axis('equal')  # optional: keeps aspect ratio square
plt.show()


# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot for x1 vs x2 with A_perturbed_vals as color
sc1 = axes[0].scatter(X1, X2, c=A_perturbed_vals, cmap="viridis", edgecolors="k", alpha=0.7)
axes[0].set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"$x_2$")
axes[0].set_title(r"Effect of $A_{01}$ on $x_1$ and $x_2$")
fig.colorbar(sc1, ax=axes[0], label=r"$A_{01}$")

# Scatter plot for x1 vs x2 with b values as color
sc2 = axes[1].scatter(X1, X2, c=b_vals, cmap="viridis", edgecolors="k", alpha=0.7)
axes[1].set_xlabel(r"$x_1$")
axes[1].set_ylabel(r"$x_2$")
axes[1].set_title(r"Effect of $b_0$ on $x_1$ and $x_2$")
fig.colorbar(sc2, ax=axes[1], label=r"$b_0$")

# Show plots
plt.tight_layout()
plt.show()
'''