# -*- coding: utf-8 -*-
"""
This is part 1 of the numerical experiments' main file. We have two settings
for the experiments:
- set1: fully deterministic case with coupling constraints
- set2: the same setting as in Lan's paper.
"""

#%% Initialization
import os
os.system('cls')  # Clear console (Windows only)
import numpy as np
import time
from FCVI import algs  # Custom algorithm implementations
import matplotlib.pyplot as plt
import cvxpy as cp  # Convex optimization
import random

random.seed(10)  # For reproducibility

%matplotlib qt  # Use interactive plot window (may depend on your IDE)

"""
Problem setup:
Objective: L(x, y) = 0.5 * x^T Q x + x^T(c + y), where Q is semi-definite
Constraints: 
- norm(x + y) <= c_{x,y}
- norm(y) <= D_y
Operator: F(x, y) = [Qx + c + y; x]
"""

tstart = time.time()  # Start timing

# Experiment parameters
R = 10  # Number of replications
K = 1000000  # Number of iterations
n = 100  # Dimension of variables x and y
m = 10  # Number of constraints in setting 2

# Arrays to store results for each algorithm and replication
time_adopex = np.zeros(R)
time_popex = np.zeros(R)
time_opconex = np.zeros(R)

tresh_eval = K / 100  # Evaluation frequency

optim_gap_adopex = np.zeros((R, int(K / tresh_eval)))
optim_gap_popex = np.zeros((R, int(K / tresh_eval)))
optim_gap_opconex = np.zeros((R, int(K / tresh_eval)))

feasib_gap_adopex = np.zeros((R, int(K / tresh_eval)))
feasib_gap_popex = np.zeros((R, int(K / tresh_eval)))
feasib_gap_opconex = np.zeros((R, int(K / tresh_eval)))

iteration_time_adopex = np.zeros((R, int(K / tresh_eval)))
iteration_time_popex = np.zeros((R, int(K / tresh_eval)))
iteration_time_opconex = np.zeros((R, int(K / tresh_eval)))

# Flags to activate algorithms
adopex_flag = 1
popex_flag = 1
opconex_flag = 1

#%% Main replication loop
for r in range(1, R + 1):
    # Generate symmetric positive semi-definite matrix Q
    diagonal_elements = np.random.uniform(1, 2, n)
    diagonal = np.diag(diagonal_elements)
    A_rand = np.random.randn(n, n)
    Q_orth, R = np.linalg.qr(A_rand)
    Q = (Q_orth.T @ diagonal) @ Q_orth
    
    c = 6 * np.ones(n)  # Linear term
    setting = 'set1'  # Change to 'set2' for Lan's paper setup
    D_x = 10
    c_xy = 1
    D_y = 3

    # Constraint matrix and noise
    V = np.random.uniform(0, 1, size=(m, n))
    beta_hat = np.zeros(m)
    xi = np.random.randn(2*K, m)
    xi_test = np.random.randn(min(5000000, 10*K), m)
    xi_test_avg = np.mean(xi_test, axis=0)
    xi_test_square = np.mean(xi_test**2, axis=0)

    # Initial values and ground-truth solution
    x_star_uncon = np.zeros(n)
    x_star = np.zeros(n)
    y_star = np.zeros(n)
    x_til = np.random.uniform(0, 1, size=(n, m))

    # Solve unconstrained version (only relevant in set2)
    if np.linalg.norm(c) > D_y:
        x_sol_uncon = cp.Variable(n)
        constraints = [cp.norm(x_sol_uncon, 2) <= D_x]
        objective = cp.Minimize(0.5 * cp.quad_form(x_sol_uncon, Q) + c.T @ x_sol_uncon + D_y * cp.norm(x_sol_uncon, 2))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        x_star_uncon = x_sol_uncon.value
        y_star_uncon = D_y * x_star_uncon / np.linalg.norm(x_star_uncon)

    # Compute constrained solution for setting 2
    if setting == 'set2':
        expectation = np.sum(((x_star_uncon[:, None] - x_til) * V.T) ** 2, axis=0) \
            + 2 * np.sum((x_star_uncon[:, None] - x_til) * V.T, axis=0) * xi_test_avg + xi_test_square
        beta_hat = 0.4 * expectation

        x_sol = cp.Variable(n)
        constraints = []
        constraints.append(cp.sum(cp.multiply(x_sol[:, None] - x_til, V.T), axis=0) ** 2
                           + 2 * cp.multiply(cp.sum(cp.multiply(x_sol[:, None] - x_til, V.T), axis=0), xi_test_avg)
                           + xi_test_square <= beta_hat)
        constraints.append(cp.norm(x_sol, 2) <= D_x)
        objective = cp.Minimize(0.5 * cp.quad_form(x_sol, Q) + c.T @ x_sol + D_y * cp.norm(x_sol, 2))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        x_star = x_sol.value
        y_star = D_y * x_star / np.linalg.norm(x_star)

        # Evaluate constraint violations
        constraint_violation = np.sum(((x_star[:, None] - x_til) * V.T) ** 2, axis=0) \
            + 2 * np.sum((x_star[:, None] - x_til) * V.T, axis=0) * xi_test_avg + xi_test_square - beta_hat

    # Build block operator H = [[Q, I]; [-I, 0]]
    H = np.block([[Q, np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    L = np.linalg.norm(H)

    # Allocate space for Lagrange multipliers' norms
    lamda_norm_adopex = np.zeros(int(K / tresh_eval))
    lamda_norm_popex = np.zeros(int(K / tresh_eval))
    lamda_norm_opconex = np.zeros(int(K / tresh_eval))

    # Compute Lipschitz constants
    if setting == 'set1':
        A_g = 2 * np.block([[np.eye(n), np.eye(n)], [np.eye(n), np.eye(n)]])
        L_g = np.linalg.norm(A_g)
        M_g = 4 * (D_x + D_y)
    if setting == 'set2':
        L_g_con = np.zeros(m)
        M_g_con = np.zeros(m)
        for j in range(m):
            L_g_con[j] = 2 * np.linalg.norm(np.outer(V[j, :], V[j, :]))
            M_g_con[j] = 2 * D_x * V[j, :].T @ V[j, :] + 2 * np.linalg.norm(V[j, :]) * np.mean(xi_test[:, :])
        L_g = np.linalg.norm(L_g_con)
        M_g = np.linalg.norm(M_g_con)

    # Initial feasible points
    if setting == 'set1':
        x_init = min(D_x, D_y) / np.sqrt(n) * np.ones(n)
        y_init = x_init
    if setting == 'set2':
        x_init = 10 * np.ones(n) * min(1, D_x / np.linalg.norm(np.ones(n)))
        y_init = D_y * x_init / np.linalg.norm(x_init)

    #%% Run algorithms
    if setting == 'set1' and adopex_flag == 1:
        tstart_adopex = time.time()
        optim_gap_adopex[r-1, :], feasib_gap_adopex[r-1, :], iteration_time_adopex[r-1, :], lamda_norm_adopex = \
            algs.adopex(n, K, Q, c, D_x, D_y, c_xy, x_init, y_init, L, L_g, M_g, tresh_eval)
        time_adopex[r-1] = time.time() - tstart_adopex
        avg_iteration_time_adopex = np.mean(np.cumsum(iteration_time_adopex, axis=1), axis=0)
        avg_opt_adopex = np.mean(optim_gap_adopex, axis=0)
        avg_feasib_adopex = np.mean(feasib_gap_adopex, axis=0)

    if popex_flag == 1:
        tstart_popex = time.time()
        optim_gap_popex[r-1, :], feasib_gap_popex[r-1, :], iteration_time_popex[r-1, :], lamda_norm_popex = \
            algs.popex(n, m, K, Q, c, D_x, D_y, c_xy, x_init, y_init, xi, L, L_g, M_g,
                       beta_hat, setting, V, tresh_eval, x_star, y_star, x_til, xi_test_avg, xi_test_square)
        time_popex[r-1] = time.time() - tstart_popex
        avg_iteration_time_popex = np.mean(np.cumsum(iteration_time_popex, axis=1), axis=0)
        avg_opt_popex = np.mean(optim_gap_popex, axis=0)
        avg_feasib_popex = np.mean(feasib_gap_popex, axis=0)

    if opconex_flag == 1:
        tstart_opconex = time.time()
        optim_gap_opconex[r-1, :], feasib_gap_opconex[r-1, :], iteration_time_opconex[r-1, :], lamda_norm_opconex = \
            algs.opconex(n, m, K, Q, c, D_x, D_y, c_xy, x_init, y_init, xi, L, L_g, M_g,
                        beta_hat, setting, V, tresh_eval, x_star, y_star, x_til, xi_test_avg, xi_test_square)
        time_opconex[r-1] = time.time() - tstart_opconex
        avg_iteration_time_opconex = np.mean(np.cumsum(iteration_time_opconex, axis=1), axis=0)
        avg_opt_opconex = np.mean(optim_gap_opconex, axis=0)
        avg_feasib_opconex = np.mean(feasib_gap_opconex, axis=0)

#%% Plot results
# For setting 1: plot all three algorithms
if setting == 'set1':
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    title_font = {'fontsize': 14, 'fontweight': 'bold', 'fontname': 'serif'}
    label_font = {'fontsize': 12, 'fontstyle': 'italic'}

    ax[0].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 1) + 1), avg_opt_adopex[1:], label="adopex", color="blue", linewidth=2, linestyle='-.')
    ax[0].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 1) + 1), avg_opt_popex[1:], label="popex", color="red", linewidth=2)
    ax[0].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 1) + 1), avg_opt_opconex[1:], label="opconex", color="black", linewidth=2, linestyle='--')
    ax[0].set_title("Optimality Gap", fontdict=title_font)
    ax[0].set_xlabel("Iteration", fontdict=label_font)
    ax[0].set_ylabel(r"$\max_{z=x,y} \langle F, (\bar{z}_k - z)\rangle$", fontweight='bold')
    ax[0].legend()

    ax[1].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 2) + 2), avg_feasib_adopex[2:], label="adopex", color="blue", linewidth=2, linestyle='-.')
    ax[1].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 2) + 2), avg_feasib_popex[2:], label="popex", color="red", linewidth=2)
    ax[1].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 2) + 2), avg_feasib_opconex[2:], label="opconex", color="black", linewidth=2, linestyle='--')
    ax[1].set_title("Feasibility Gap", fontdict=title_font)
    ax[1].set_xlabel("Iteration", fontdict=label_font)
    ax[1].set_ylabel(r"$\left\| [g(\bar{z}_k)]_+ \right\|$", fontweight='bold')
    ax[1].legend()
    plt.tight_layout()

# For setting 2: plot only popex and opconex
if setting == 'set2':
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    title_font = {'fontsize': 14, 'fontweight': 'bold', 'fontname': 'serif'}
    label_font = {'fontsize': 12, 'fontstyle': 'italic'}

    if popex_flag == 1:
        ax[0].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 1) + 1), avg_opt_popex[1:], label="popex", color="red", linewidth=2)
    if opconex_flag == 1:
        ax[0].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 1) + 1), avg_opt_opconex[1:], label="opconex", color="black", linewidth=2, linestyle='--')
    ax[0].set_title("Optimality Gap", fontdict=title_font)
    ax[0].set_xlabel("Iteration", fontdict=label_font)
    ax[0].set_ylabel(r"$ \langle F, (\bar{z}_k - z^*)\rangle$", fontweight='bold')
    ax[0].legend()

    if popex_flag == 1:
        ax[1].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 2) + 2), avg_feasib_popex[2:], label="popex", color="red", linewidth=2)
    if opconex_flag == 1:
        ax[1].plot(tresh_eval * (np.arange(int(K / tresh_eval) - 2) + 2), avg_feasib_opconex[2:], label="opconex", color="black", linewidth=2, linestyle='--')
    ax[1].set_title("Feasibility Gap", fontdict=title_font)
    ax[1].set_xlabel("Iteration", fontdict=label_font)
    ax[1].set_ylabel(r"$\left\| [g(\bar{z}_k)]_+ \right\|$")
    ax[1].legend()
    plt.tight_layout()
