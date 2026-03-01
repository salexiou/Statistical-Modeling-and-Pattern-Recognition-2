import numpy as np

# Data for class ω1
data_omega_1 = np.array([
    [0.42, -0.87, 0.58],
    [-0.2, -3.3, -3.4],
    [1.3, -0.32, 1.7],
    [0.39, 0.71, 0.23],
    [-1.6, -5.3, -0.15],
    [-0.029, 0.89, -4.7],
    [-0.23, 1.9, 2.2],
    [0.27, -0.3, -0.87],
    [-1.9, 0.76, -2.1],
    [0.87, -1, -2.6]
])

# Data for class ω2
data_omega_2 = np.array([
    [-0.4, 0.58, 0.089],
    [-0.31, 0.27, -0.04],
    [0.38, 0.055, -0.035],
    [-0.15, 0.53, 0.011],
    [-0.35, 0.47, 0.034],
    [0.17, 0.69, 0.1],
    [-0.011, 0.55, -0.18],
    [-0.27, 0.61, 0.12],
    [-0.065, 0.49, 0.0012],
    [-0.12, 0.054, -0.063]
])

# ------------------------------------ a) ------------------------------------ #
print("\n ------------------------ a ------------------------")
mu_hat = np.mean(data_omega_1, axis=0)
sigma_hat_squared = np.var(data_omega_1, axis=0,ddof= 0)

for i in range(3):
    print(f"Feature x_{i+1}:")
    print(f"  Mean: {mu_hat[i]}")
    print(f"  Variance: {sigma_hat_squared[i]}\n")
# ---------------------------------------------------------------------------- #

# ------------------------------------ b) ------------------------------------ #
print("\n ------------------------ b ------------------------")
def calculate_2d_normal_params(data):
    mu = np.mean(data, axis=0)
    
    sigma = np.cov(data, rowvar=False,bias = True)
    
    return mu, sigma

pairs = [(0, 1), (0, 2), (1, 2)]

for i, (index1, index2) in enumerate(pairs):
    data_pair = data_omega_1[:, [index1, index2]]
    mu, sigma = calculate_2d_normal_params(data_pair)
    
    print(f"Pair {i+1} (x_{index1+1}, x_{index2+1}):")
    print(f"  Mean vector: {mu}")
    print(f"  Covariance matrix:\n{sigma}\n")
# ---------------------------------------------------------------------------- #


# ------------------------------------ c) ------------------------------------ #
print("\n ------------------------ c ------------------------")
def calculate_3d_normal_params(data):
    mu = np.mean(data, axis=0)
    
    sigma = np.cov(data, rowvar=False,bias = True)
    
    return mu, sigma

mu, sigma = calculate_3d_normal_params(data_omega_1)

print(f"Mean vector : {mu}")
print(f"Covariance matrix :\n{sigma}")
# ---------------------------------------------------------------------------- #

# ------------------------------------ d) ------------------------------------ #
print("\n ------------------------ d ------------------------")

def calculate_diagonal_normal_params(data):
    mu_hat = np.mean(data, axis=0)
    sigma_hat_squared = np.var(data, axis=0, ddof=0)  
    
    diagonal_cov_matrix = np.diag(sigma_hat_squared)
    
    return mu_hat, diagonal_cov_matrix

mu_hat, diagonal_cov_matrix = calculate_diagonal_normal_params(data_omega_2)

print(f"Mean vector: {mu_hat}")
print(f"Diagonal covariance matrix:\n{diagonal_cov_matrix}")

# ---------------------------------------------------------------------------- #
