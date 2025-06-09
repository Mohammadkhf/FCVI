# -*- coding: utf-8 -*-
""" 
This is part two of the numerical experiment section, involving data-driven Robust Optimization.

The problem has the form:
min_{p} max_{theta_hat} E[p * D(s, p, xi; theta_hat)],  
subject to: E[p * D(tildes_i, p_i, zeta_i; theta_hat)] >= d_i
"""

#%% Import libraries and clear console
import os
os.system('cls')  # Clear the console (Windows specific)
import numpy as np
import time
#import FCVI  # Commented out, maybe for future use
from FCVI_pricing import part2  # Import part2 module with algorithms
#import matplotlib as mtplot  # Commented out
import matplotlib.pyplot as plt
import cvxpy as cp  # Convex optimization library
#import dsp  # Commented out, possibly for advanced optimization functions
from scipy.optimize import minimize_scalar
import random

random.seed(10)  # Set random seed for reproducibility

# Enable interactive plotting with Qt backend
%matplotlib qt  

#%% Experiment settings and pre-allocate arrays for results

R = 10  # Number of replications
K = 500000  # Number of iterations per replication
tresh_eval = int(K / 100)  # Evaluation interval for metrics (every 1% of iterations)

# Pre-allocate arrays to store results for each algorithm and replication
time_opconex = np.zeros(R)
optim_gap_opconex = np.zeros((R, int(K / tresh_eval)))
feasib_gap_opconex = np.zeros((R, int(K / tresh_eval)))
iteration_time_opconex = np.zeros((R, int(K / tresh_eval)))

time_popex = np.zeros(R)
optim_gap_popex = np.zeros((R, int(K / tresh_eval)))
feasib_gap_popex = np.zeros((R, int(K / tresh_eval)))
iteration_time_popex = np.zeros((R, int(K / tresh_eval)))

time_RLSA = np.zeros(R)
optim_gap_RLSA = np.zeros((R, int(K / tresh_eval)))
feasib_gap_RLSA = np.zeros((R, int(K / tresh_eval)))
iteration_time_RLSA = np.zeros((R, int(K / tresh_eval)))

#%% Main loop over replications
for r in range(1, R + 1):
    
    # Problem dimensions and parameters
    d = 20  # Dimension of theta vector
    m = 100  # Number of demand samples
    
    # Generate test data xi for evaluation
    xi_test = np.random.randn(10 * K, m)  # Normal random variables for testing
    xi_test_avg = np.mean(xi_test, axis=0)  # Mean of xi_test over samples
    xi_test_square = np.mean(xi_test**2, axis=0)  # Second moment of xi_test
    
    # Define bounds for theta and theta_0
    l = np.random.uniform(1, 3, size=(d))  # Lower bound for theta
    l_0 = -5  # Lower bound for scalar theta_0
    omega = np.random.uniform(1, 4, size=(d))  # Additional range for upper bound
    u = l + omega  # Upper bound for theta
    u_0 = l_0 + np.random.uniform(1, 4)  # Upper bound for theta_0
    
    p_max = 20  # Maximum price p
    
    # Generate random demand parameters
    s = np.random.uniform(0, 3, d)  # Vector s
    s_tild = np.random.uniform(0, 3, size=(m, d))  # Matrix s_tild with m samples
    p_tild = np.random.uniform(10, 20, size=m)  # Vector of prices for demand samples
    
    # Initialize theta_tild with random values within bounds
    theta_tild = np.zeros(d)
    for i in range(d):
        theta_tild[i] = np.random.uniform(l[i], u[i])
    theta_tild_0 = np.random.uniform(l_0, u_0)  # Scalar theta_0 for demand
    
    # Compute demand as max(0, linear function)
    demand = np.maximum((s_tild @ theta_tild) + theta_tild_0 * p_tild, 0)
    
    # Generate xi noise vector
    xi = np.random.randn(2 * K)
    
    # Define function D(s, p, theta, theta_0, xi) - evaluation function
    def D(s_f, p_f, theta_f, theta_0_f, xi_f):
        return s_f @ theta_f + theta_0_f * p_f + np.mean(xi_f)
    
    # Solve the inner optimization problem for fixed p
    def solve_inner_problem(p_val):
        theta = cp.Variable(d)  # Optimization variable theta
        theta_0 = cp.Variable()  # Optimization variable scalar theta_0
        
        # Constraints to ensure demand satisfaction and bounds on parameters
        constraints = [
            s_tild @ theta + theta_0 * p_tild >= demand,  # Demand constraints
            theta >= l,  # Lower bound on theta
            theta <= u,  # Upper bound on theta
            theta_0 >= l_0,  # Lower bound on theta_0
            theta_0 <= u_0   # Upper bound on theta_0 (concavity in p)
        ]
        
        # Objective: minimize expected cost function for fixed p_val
        obj = cp.Minimize(s @ theta + theta_0 * p_val + np.mean(xi))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        
        return theta.value, theta_0.value, prob.value
    
    # Define outer problem objective (maximize wrt p, minimized by minimize_scalar)
    def outer_objective(p_val):
        if not (0 <= p_val <= p_max):
            return 1e10  # Return large penalty if p_val out of bounds
        theta_star, theta_0_star, inner_val = solve_inner_problem(p_val)
        return -p_val * inner_val  # maximize by minimizing negative
    
    # Solve outer problem over p in [0, p_max]
    result = minimize_scalar(outer_objective, bounds=(0, p_max), method='bounded')
    p_star = result.x  # Optimal price
    outer_prob_value = -result.fun  # Optimal objective value
    
    # Get corresponding theta and theta_0 at optimal p
    theta_star, theta_0_star, inner_prob_value = solve_inner_problem(p_star)
    
    # Commented out code below was likely used for other problem formulations
    # or debugging and can be revisited later
    
    # Calculate variance and Lipschitz constants for problem parameters
    sample_variance_xi = np.var(xi, ddof=1)
    L = u_0 + np.sqrt(u_0**2 + np.linalg.norm(s)**2 + 4 * p_max**2)  # Lipschitz constant
    var_obj = p_max**2 * sample_variance_xi
    var_g = 0 
    var_g_grad = 0 
    L_g = 0 
    
    # Compute norm bounds for constraints gradients M_g
    M_g_cons = np.zeros(m)
    for j in range(m):
        M_g_cons[j] = np.linalg.norm(s_tild[j, :]) + p_tild[j]
    M_g = np.linalg.norm(M_g_cons)
    
    B = 100  # Some constant used in algorithms
    
    # Initialize variables for algorithms
    p_init = 0
    theta_init = np.ones(d)
    theta_0_init = -1
    
    #%% Run the OPConex algorithm and record performance metrics
    tstart_opconex = time.time()
    optim_gap_opconex[r-1, 0:int(K / tresh_eval)], feasib_gap_opconex[r-1, 0:int(K / tresh_eval)], iteration_time_opconex[r-1, 0:int(K / tresh_eval)], lamda_norm_opconex = part2.opconex(
        d, m, K, l, l_0, u, u_0, p_max, p_init, theta_init, theta_0_init, B, xi, L, L_g, M_g, var_obj, var_g, var_g_grad, tresh_eval, p_star, theta_star, theta_0_star, xi_test_avg, xi_test_square, s, s_tild, theta_tild, theta_tild_0, demand, p_tild
    )
    time_opconex[r-1] = time.time() - tstart_opconex
    
    # Compute average iteration time and gaps over replications so far
    avg_iteration_time_opconex = np.mean(np.cumsum(iteration_time_opconex, axis=1), axis=0)
    avg_opt_opconex = np.mean(optim_gap_opconex, axis=0)
    avg_feasib_opconex = np.mean(feasib_gap_opconex, axis=0)
    
    '''
    # Code for running the Popex algorithm is commented out for now
    tstart_popex = time.time()
    optim_gap_popex[r-1, 0:int(K / tresh_eval)], feasib_gap_popex[r-1, 0:int(K / tresh_eval)], iteration_time_popex[r-1, 0:int(K / tresh_eval)], lamda_norm_popex = part2.popex(
        d, m, K, l, l_0, u, u_0, p_max, p_init, theta_init, theta_0_init, B, xi, L, L_g, M_g, var_obj, var_g, var_g_grad, tresh_eval, p_star, theta_star, theta_0_star, xi_test_avg, xi_test_square, s, s_tild, theta_tild, theta_tild_0, demand, p_tild
    )
    time_popex[r-1] = time.time() - tstart_popex
    avg_iteration_time_popex = np.mean(np.cumsum(iteration_time_popex, axis=1), axis=0)
    avg_opt_popex = np.mean(optim_gap_popex, axis=0)
    avg_feasib_popex = np.mean(feasib_gap_popex, axis=0)
    '''
    
    # Run the RLSA algorithm and record performance metrics
    tstart_RLSA = time.time()
    optim_gap_RLSA[r-1, 0:int(K / tresh_eval)], feasib_gap_RLSA[r-1, 0:int(K / tresh_eval)], iteration_time_RLSA[r-1, 0:int(K / tresh_eval)], lamda_norm_RLSA = part2.RLSA(
        d, m, K, l, l_0, u, u_0, p_max, p_init, theta_init, theta_0_init, B, xi, L, L_g, M_g, var_obj, var_g, var_g_grad, tresh_eval, p_star, theta_star, theta_0_star, xi_test_avg, xi_test_square, s, s_tild, theta_tild, theta_tild_0, demand, p_tild
    )
    time_RLSA[r-1] = time.time() - tstart_RLSA
    
    # Compute averages for RLSA
    avg_iteration_time_RLSA = np.mean(np.cumsum(iteration_time_RLSA, axis=1), axis=0)
    avg_opt_RLSA = np.mean(optim_gap_RLSA, axis=0)
    avg_feasib_RLSA = np.mean(feasib_gap_RLSA, axis=0)

#%% Plot results: Optimality and Feasibility Gaps

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

title_font = {'fontsize': 14, 'fontweight': 'bold', 'fontname': 'serif'}
label_font = {'fontsize': 12, 'fontstyle': 'italic'}

# Plot Optimality Gap
ax[0].plot(
    tresh_eval * (np.arange(int(K / tresh_eval) - 10) + 10),
    avg_opt_opconex[10:],
    label="opconex",
    color="black",
    linewidth=2,
    linestyle='--'
)
ax[0].plot(
    tresh_eval * (np.arange(int(K / tresh_eval) - 10) + 10),
    avg_opt_RLSA[10:],
    label="RLSA",
    color="blue",
    linewidth=2
)
#ax[0].plot(tresh_eval * np.arange(int(K / tresh_eval)), avg_opt_popex, label="popex", color="red", linewidth=2)  # commented out
ax[0].set_title("Optimality Gap", fontdict=title_font)
ax[0].set_xlabel("Iteration", fontdict=label_font)
ax[0].set_ylabel(r"$ \langle F, (\bar{z}_k - z^*)\rangle$", fontweight='bold')
ax[0].legend()

# Plot Feasibility Gap
ax[1].plot(
    tresh_eval * (np.arange(int(K / tresh_eval) - 2) + 2),
    avg_feasib_opconex[2:],
    label="opconex",
    color="black",
    linewidth=2,
    linestyle='--'
)
ax[1].plot(
    tresh_eval * (np.arange(int(K / tresh_eval) - 2) + 2),
    avg_feasib_RLSA[2:],
    label="RLSA",
    color="blue",
    linewidth=2
)
#ax[1].plot(tresh_eval * np.arange(int(K / tresh_eval)), avg_feasib_popex, label="popex", color="red", linewidth=2)  # commented out
ax[1].set_title("Feasibility Gap", fontdict=title_font)
ax[1].set_xlabel("Iteration", fontdict=label_font)
ax[1].set_ylabel(r"$\left\| [g(\bar{z}_k)]_+ \right\|$")
ax[1].legend()

plt.tight_layout()

