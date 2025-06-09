# -*- coding: utf-8 -*-
"""
AUC maximization with fairness constraints 
"""

#%% Importing libraries 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
os.system('cls')  # Clears terminal screen (for Windows)
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
import time
import matplotlib.pyplot as plt
import cvxpy as cp
import random 

from AUC import alg  # Custom module containing algorithm

# Set random seed for reproducibility
random.seed(10)

# Enable interactive plotting window
%matplotlib qt 

"""
Problem setup:
AUC maximization with fairness constraints on the Adult Income dataset.
Sensitive attribute: gender.
"""

# Load dataset
df = pd.read_csv('adult.csv')
df.replace('?', np.nan, inplace=True)

# Continuous columns to log-transform (for normalization)
continuous_cols = ['fnlwgt','capital-gain', 'capital-loss']
df[continuous_cols] = np.log(df[continuous_cols]+1)

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# One-hot encode categorical columns, drop first to avoid multicollinearity
df1 = pd.get_dummies(
    df,
    columns=['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'],
    drop_first=True,
    dtype=float
)

# Convert income labels to -1 and 1
df1['income'] = (
    df1['income']
    .replace({'<=50K': -1, '>50K': 1})
    .infer_objects(copy=False)
    .astype(int)
)

# Drop 'education' column (not needed)
df1.drop(columns='education', inplace=True)

# Prepare features and response
df1_X = df1.drop(['income'], axis=1)
N = df1.shape[0]
N_pos = df1[df1['income']==1].shape[0]
N_neg = df1[df1['income']==-1].shape[0]
p = N_pos / N
y_responce = df1['income']

# Separate positive and negative samples
df_positive = df1[df1['income'] == 1]
df_negative = df1[df1['income'] == -1]
df_pos_X = df_positive.drop(['income'], axis=1)
df_neg_X = df_negative.drop(['income'], axis=1)

# Convert to numpy arrays
X_pos = df_pos_X.to_numpy()
X_neg = df_neg_X.to_numpy()
X = df1_X.to_numpy()
y_responce_np = y_responce.to_numpy()

# Compute empirical statistics needed for objective
w_tran_w_pos = np.sum(np.einsum('ij,ij->i', X_pos, X_pos))
w_tran_w_neg = np.sum(np.einsum('ij,ij->i', X_neg, X_neg))
w_sum_pos = np.sum(X_pos, axis=0)
w_sum_neg = np.sum(X_neg, axis=0)

# Fairness constraint setup
u_bar = df1_X['gender_Male'].mean()
u = df1_X['gender_Male'].values
u_u_bar_w = (u - u_bar) @ X
emp_expc_fair = u_u_bar_w / N

# Constraint and model parameters
c = 0.02
D_x = 10
a_max = 10
b_max = 10 

# Optimization variables
d = df1_X.shape[1]  # number of features
x_sol = cp.Variable(d)
a_sol = cp.Variable()
b_sol = cp.Variable()

# Define AUC loss components
x_pos_a = (1-p)*cp.sum_squares(X_pos @ x_sol - a_sol)
x_neg_b = p*cp.sum_squares(X_neg @ x_sol - b_sol)
V = (p * w_sum_neg - (1 - p) * w_sum_pos)
x_w_pos_w_neg = 2 * x_sol @ V
square_x_pos_w_neg = cp.square(x_sol @ V) / (p*(1-p)*N)

# Final empirical objective
emp_exp = (x_pos_a + x_neg_b + x_w_pos_w_neg + square_x_pos_w_neg) / N
obj = cp.Minimize(emp_exp)

# Fairness and box constraints
constraints = [
    (emp_expc_fair @ x_sol) <= c,
    (emp_expc_fair @ x_sol) >= -c,
    cp.norm(x_sol) <= D_x,
    a_sol <= a_max,
    b_sol <= b_max
]

# Solve convex optimization problem using MOSEK
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.MOSEK)

# Extract optimal values
optimal_val = prob.value
x_star = x_sol.value
a_star = a_sol.value
b_star = b_sol.value
alpha_star = (V @ x_star)/(p*(1-p)*N)
alpha_max = alpha_star + 1  # buffer for initialization

# Initialization for algorithm
x_init = np.ones(d)/np.sqrt(d) * D_x / 2
a_init = a_max / 2
b_init = b_max / 2
alpha_init = 0

# Compute constants for algorithm (Lipschitz, variance bounds)
M_g = np.linalg.norm(emp_expc_fair)
L_g = 0

# Compute full Hessian matrix for Lipschitz constant
Sigma_plus = X_pos.T @ X_pos
Sigma_minus = X_neg.T @ X_neg
mu_plus = np.sum(X_pos, axis=0)
mu_minus = np.sum(X_neg, axis=0)

H_xx = 2 * ((1 - p) * Sigma_plus + p * Sigma_minus) / N
H_xa = -2 * (1 - p) * mu_plus[:, None] / N
H_xb = -2 * p * mu_minus[:, None] / N
H_xalpha = 2 * (p * mu_minus - (1 - p) * mu_plus)[:, None] / N

H_ax = H_xa.T
H_aa = np.array([[2 * (1 - p) * N_pos / N]])
H_ab = np.array([[0]])
H_aalpha = np.array([[0]])

H_bx = H_xb.T
H_ba = np.array([[0]])
H_bb = np.array([[2 * p * N_neg / N]])
H_balpha = np.array([[0]])

H_alphax = H_xalpha.T
H_alphaa = np.array([[0]])
H_alphab = np.array([[0]])
H_alphaalpha = np.array([[2 * p * (1 - p)]])

# Stack to form full Hessian
top = np.hstack([H_xx, H_xa, H_xb, H_xalpha])
row_a = np.hstack([H_ax, H_aa, H_ab, H_aalpha])
row_b = np.hstack([H_bx, H_ba, H_bb, H_balpha])
row_alpha = np.hstack([H_alphax, H_alphaa, H_alphab, H_alphaalpha])
H = np.vstack([top, row_a, row_b, row_alpha])
L_F = np.linalg.norm(H)

# Variance estimation for stochastic approximation
R = np.max(np.linalg.norm(X, axis=1))

gx_bound = (2 * (1 - p) * (D_x * R + a_max) * R +
            2 * p * (D_x * R + b_max) * R +
            2 * (1 + alpha_max) * R)

ga_bound = 2 * (1 - p) * (D_x * R + a_max)
gb_bound = 2 * p * (D_x * R + b_max)
galpha_bound = 2 * D_x * R + 2 * p * (1 - p) * alpha_max

sigma_squared_f = gx_bound**2 + ga_bound**2 + gb_bound**2 + galpha_bound**2
sigma_f = np.sqrt(sigma_squared_f)

sigma_g = 2 * D_x * R
sigma_grad_g = np.sqrt(2) * R

# Algorithm settings
M = 100              # Minibatch size
rep = 5              # Number of replications
K = int(100 * N / M) # Total number of iterations for 100 data passes
tresh_eval = int(K / 100)

# Arrays to store metrics
time_opconex = np.zeros(rep)
optim_gap_opconex = np.zeros((rep, int(K / tresh_eval) + 1))
feasib_gap_opconex = np.zeros((rep, int(K / tresh_eval) + 1))
iteration_time_opconex = np.zeros((rep, int(K / tresh_eval) + 1))

flag = 'minibatch'

#%% Run the algorithm
for r in range(rep):
    tstart_opconex = time.time()
    optim_gap, feasib_gap, iteration_time = alg.opconex(
        x_star, a_star, b_star, alpha_star,
        D_x, a_max, b_max, alpha_max,
        M_g, L_g, L_F, c,
        x_init, a_init, b_init, alpha_init,
        p, sigma_g, sigma_f, sigma_grad_g, K,
        emp_expc_fair, tresh_eval, X, y_responce_np, d, M, N, flag
    )
    optim_gap_opconex[r, :] = optim_gap
    feasib_gap_opconex[r, :] = feasib_gap
    iteration_time_opconex[r, :] = iteration_time
    time_opconex[r-1] = time.time() - tstart_opconex
    avg_iteration_time_opconex = np.mean(np.cumsum(iteration_time_opconex, axis=1), axis=0)
    avg_opt_opconex = np.mean(optim_gap_opconex, axis=0)
    avg_feasib_opconex = np.mean(feasib_gap_opconex, axis=0)
    print(f" replication {r+1} is done...")

#%% Plot results
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
title_font = {'fontsize': 14, 'fontweight': 'bold', 'fontname': 'serif'}
label_font = {'fontsize': 12, 'fontstyle': 'italic'}

# Plot optimality gap
ax[0].plot(np.arange(int(K/tresh_eval)-3)+3, np.abs(avg_opt_opconex[4:]),
           label="opconex", color="black", linewidth=2)
ax[0].plot(np.arange(int(K/tresh_eval)-4)+4, np.abs(avg_opt_deter_eta_10k_tau_6k[4:]),
           label="opconex-deterministic", color="red", linewidth=2, linestyle='--')
ax[0].set_title("Optimality Gap", fontdict=title_font)
ax[0].set_xlabel("Data Passes", fontdict=label_font)
ax[0].set_ylabel(r"$\langle F, (\bar{z}_k - z^*)\rangle$")
ax[0].legend()

# Plot feasibility gap
ax[1].plot(np.arange(int(K/tresh_eval)-0)+0, avg_feasib_opconex[1:],
           label="opconex", color="black", linewidth=2)
ax[1].plot(np.arange(int(K/tresh_eval)-1)+1, np.abs(avg_feas_deter_eta_10k_tau_6k[1:]),
           label="opconex-deterministic", color="red", linewidth=2, linestyle='--')
ax[1].set_title("Feasibility Gap", fontdict=title_font)
ax[1].set_xlabel("Data Passes", fontdict=label_font)
ax[1].set_ylabel(r"$\left\| [g(\bar{x}_k)]_+ \right\|$")
ax[1].legend()

plt.tight_layout()

