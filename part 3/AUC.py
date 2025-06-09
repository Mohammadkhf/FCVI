# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:53:02 2025

@author: Mohamad
"""

import numpy as np
import time as tp
# import cvxpy as cp (currently unused)

# Function to evaluate the two-sided fairness constraints
def constraints(a, x, c):
    """
    Compute the fairness constraints 
    
    Args:
        a : vector in |a^T x|<=c 
        x : decision variable 
        c : threshold 
    Returns:
        g : vector containing both sides of the constraint
    """
    g_right  =  a @ x - c 
    g_left = -a @ x - c
    g = np.array([g_right, g_left])
    return g

# Function to compute the operator F using a minibatch
def operator_F_minibatch(x, a, b, alpha, W_batch, y_batch, p):
    """
    Randomly sample a minibatch and compute the average operator F.

    Args:
        x : current iterate (decision variable)
        a, b, alpha : scalar parameters
        W_batch : minibatch of data samples
        y_batch : corresponding labels
        p : class prior (P(y = -1))

    Returns:
        A list of four averaged components of the operator F
    """
    I_pos = (y_batch == 1).astype(float)
    I_neg = (y_batch == -1).astype(float)

    x_dot_w = W_batch @ x  # shape (M,)

    # Compute F_x for each sample in batch
    F_x_batch = (
        2 * (1 - p) * ((x_dot_w - a)[:, None] * W_batch) * I_pos[:, None] +
        2 * p * ((x_dot_w - b)[:, None] * W_batch) * I_neg[:, None] +
        2 * (1 + alpha) * (p * W_batch * I_neg[:, None] - (1 - p) * W_batch * I_pos[:, None])
    )

    # Compute scalar parts of F
    F_a_batch = -2 * (1 - p) * (x_dot_w - a) * I_pos
    F_b_batch = -2 * p * (x_dot_w - b) * I_neg
    F_alpha_batch = -2 * (p * x_dot_w * I_neg - (1 - p) * x_dot_w * I_pos) + 2 * p * (1 - p) * alpha

    # Return averaged values over minibatch
    F_x_avg = np.mean(F_x_batch, axis=0)
    F_a_avg = np.mean(F_a_batch)
    F_b_avg = np.mean(F_b_batch)
    F_alpha_avg = np.mean(F_alpha_batch)

    return [F_x_avg, F_a_avg, F_b_avg, F_alpha_avg]

# Main algorithm class
class alg:
    def opconex(x_star, a_star, b_star, alpha_star,
                D_x, a_max, b_max, alpha_max,
                M_g, L_g, L_F, c,
                x_init, a_init, b_init, alpha_init,
                p, sigma_g, sigma_f, sigma_grad_g, K,
                emp_expc_fair, tresh_eval, X, y_responce_np, d, M, N, flag):
        
        # Step size and regularization constants
        theta = 1 
        B = 5
        eta = (L_g * B) + (8 * L_F + 8 * M_g * B / np.sqrt(D_x**2 + a_max**2 + b_max**2)) + \
              (2 * np.sqrt(K) * (np.sqrt(2) * sigma_f + 4 * B * sigma_grad_g) / D_x)
        tau = (9 * max(M_g, sigma_grad_g) * np.sqrt(D_x**2 + a_max**2 + b_max**2)) / B + \
              (8 / B * np.sqrt(K) * np.sqrt(sigma_f**2 + D_x**2 * sigma_grad_g**2))
        
        s = 0 
        lam_old = 0
        lam_new = 0

        # Initial constraint violation
        ell_previous = constraints(emp_expc_fair, x_init, c)
        ell_current  = ell_previous
        ell_next = ell_current

        # Initialize dual variables
        u_t_x = np.zeros(d)
        u_t_a = 0
        u_t_b = 0 
        u_t_alpha = 0 

        # Initialize primal variables
        x_previous = x_init
        x_current  = x_previous
        x_next = x_current 

        a_previous  = a_init
        a_current  = a_previous
        a_next = a_current

        b_previous  = b_init
        b_current = b_previous
        b_next = b_current

        alpha_previous = alpha_init 
        alpha_current  = alpha_previous
        alpha_next = alpha_current

        # Initialize running averages
        x_bar_cur = x_init
        x_bar_next = x_bar_cur

        a_bar_cur = a_init 
        a_bar_next = a_bar_cur

        b_bar_cur = b_init 
        b_bar_next = b_bar_cur

        alpha_bar_cur = alpha_init 
        alpha_bar_next = alpha_bar_cur

        # Logging
        optim_gap = np.zeros(int(K / tresh_eval) + 1)
        feasib_gap = np.zeros(int(K / tresh_eval) + 1)
        iteration_time = np.zeros(int(K / tresh_eval) + 1)

        # Gradients of the constraints
        grad_g__right = emp_expc_fair
        grad_g__left  = -emp_expc_fair
        grad_g = [grad_g__right, grad_g__left]

        # True solution vector (used for gap computation)
        z_star = np.concatenate([x_star, [a_star], [b_star], [alpha_star]])

        # Main loop
        for k in range(1, K + 1):
            eta = 10 * np.sqrt(k)
            tau = 5 * np.sqrt(k)

            # Log initial metrics
            if k == 1:
                z_bar_cur = np.concatenate([x_bar_cur, [a_bar_cur], [b_bar_cur], [alpha_bar_cur]])
                F_1 = operator_F_minibatch(x_bar_cur, a_bar_cur, b_bar_cur, alpha_bar_cur, X, y_responce_np, p)
                F_vector = np.concatenate([F_1[0], [F_1[1]], [F_1[2]], [F_1[3]]])
                optim_gap[k - 1] = F_vector @ (z_bar_cur - z_star)
                feasib_gap[k - 1] = np.linalg.norm(np.maximum(constraints(emp_expc_fair, x_bar_cur, c), 0))
                tStart_opconex = tp.time()

            # Periodically evaluate and log performance
            if k % tresh_eval == 0:
                iteration_time[int(k / tresh_eval)] = tp.time() - tStart_opconex
                z_bar_cur = np.concatenate([x_bar_cur, [a_bar_cur], [b_bar_cur], [alpha_bar_cur]])
                F_1 = operator_F_minibatch(x_bar_cur, a_bar_cur, b_bar_cur, alpha_bar_cur, X, y_responce_np, p)
                F_vector = np.concatenate([F_1[0], [F_1[1]], [F_1[2]], [F_1[3]]])
                optim_gap[int(k / tresh_eval)] = F_vector @ (z_bar_cur - z_star)
                feasib_gap[int(k / tresh_eval)] = np.linalg.norm(np.maximum(constraints(emp_expc_fair, x_bar_cur, c), 0))
                print(f"Iteration {k} of opconex completed...")
                tStart_opconex = tp.time()

            # Dual variable update
            s = (1 + theta) * ell_current - theta * ell_previous
            lam_new = np.maximum(lam_old + s / tau, 0)
            lam_old = lam_new

            # Sample a new minibatch if flag is set
            if flag == 'minibatch':
                idx = np.random.choice(N, M, replace=False)
                W_batch = X[idx]
                y_batch = y_responce_np[idx]
                operator_F_current = operator_F_minibatch(x_current, a_current, b_current, alpha_current, W_batch, y_batch, p)
                operator_F_previous = operator_F_minibatch(x_previous, a_previous, b_previous, alpha_previous, W_batch, y_batch, p)
            else:
                operator_F_current = operator_F_minibatch(x_current, a_current, b_current, alpha_current, X, y_responce_np, p)
                operator_F_previous = operator_F_minibatch(x_previous, a_previous, b_previous, alpha_previous, X, y_responce_np, p)

            # Gradient updates for x, a, b, alpha
            u_t_x = (1 + theta) * operator_F_current[0] - theta * operator_F_previous[0] + lam_new @ grad_g
            x_next = (x_current - u_t_x / eta) * min(1, D_x / np.linalg.norm(x_current - u_t_x / eta))

            u_t_a = (1 + theta) * operator_F_current[1] - theta * operator_F_previous[1]
            a_next = min(max(0, (a_current - u_t_a / eta)), a_max)

            u_t_b = (1 + theta) * operator_F_current[2] - theta * operator_F_previous[2]
            b_next = min(max(0, (b_current - u_t_b / eta)), b_max)

            u_t_alpha = (1 + theta) * operator_F_current[3] - theta * operator_F_previous[3]
            alpha_next = min(max(0, (alpha_current - u_t_alpha / eta)), alpha_max)

            # Evaluate new constraint value
            ell_next = constraints(emp_expc_fair, x_next, c)

            # Shift variables
            ell_previous = ell_current
            ell_current = ell_next

            x_previous = x_current
            x_current = x_next
            a_previous = a_current
            a_current = a_next
            b_previous = b_current
            b_current = b_next
            alpha_previous = alpha_current
            alpha_current = alpha_next

            # Update running average
            x_bar_next = (x_next - x_bar_cur) / k + x_bar_cur
            a_bar_next = (a_next - a_bar_cur) / k + a_bar_cur
            b_bar_next = (b_next - b_bar_cur) / k + b_bar_cur
            alpha_bar_next = (alpha_next - alpha_bar_cur) / k + alpha_bar_cur

            x_bar_cur = x_bar_next
            a_bar_cur = a_bar_next
            b_bar_cur = b_bar_next
            alpha_bar_cur = alpha_bar_next

        return optim_gap, feasib_gap, iteration_time
