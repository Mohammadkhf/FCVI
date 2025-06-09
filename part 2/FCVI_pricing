# -*- coding: utf-8 -*-
"""
FCVI methods of three main algorithms for two different sets 
"""
import numpy as np
import time as tp
#import cvxpy as cp

# Define demand function D(s, p, theta, theta_0, xi)
def D(s_f, p_f, theta_f, theta_0_f, xi_f):
    # Compute demand as a linear combination of supply, price, and noise
    return s_f @ theta_f + theta_0_f * p_f + np.mean(xi_f)

# Define constraint function D_con(s, p, theta, theta_0, demand)
def D_con(s_f, p_f, theta_f, theta_0_f, demand):
    # Compute constraint violation: demand minus target demand
    return s_f @ theta_f + theta_0_f * p_f - demand

# Define operator for price update
def p_operator(s_f, p_f, theta_f, theta_0_f, xi_f):
    # Compute gradient-like term for price update
    return -s_f @ theta_f - 2 * theta_0_f * p_f - xi_f

# Define operator for theta update
def theta_operator(s_f, p_f):
    # Compute gradient-like term for theta update based on supply and price
    return p_f * s_f

# Define operator for theta_0 update
def theta_0_operator(p_f):
    # Compute gradient-like term for theta_0 update based on price squared
    return p_f**2   

class part2:
    # Opconex method for optimization
    def opconex(d, m, K, l, l_0, u, u_0, p_max, p_init, theta_init, theta_0_init, B, xi, L, L_g, M_g, var_obj, var_g, var_g_grad, tresh_eval, p_star, theta_star, theta_0_star, xi_test_avg, xi_test_square, s, s_tild, theta_tild, theta_tild_0, demand, p_tild):
        # Initialize parameters
        theta_para = 1  # Momentum parameter for updates
        eta = (L_g * B) + (8 * L + 8 * M_g * B / p_max) + (8 * B * np.sqrt(var_g_grad)) * np.sqrt(K) / p_max  # Initial step size for p, theta, and theta_0 updates
        tau = (9 * max(M_g, np.sqrt(var_g_grad)) * p_max) / B + (8 / B * np.sqrt(K) * np.sqrt(var_g + var_g_grad))  # Initial step size for Lagrange multiplier
        s_para = 0  # Scalar for constraint violation
        lam_old = 0  # Previous Lagrange multiplier
        lam_new = 0  # Current Lagrange multiplier
        ell_previous = -D_con(s_tild, p_tild, theta_init, theta_0_init, demand)  # Initial constraint violation
        ell_current = ell_previous  # Current constraint violation
        ell_next = np.zeros(m)  # Next constraint violation
        u_t_p = 0  # Update term for price
        u_t_theta = np.zeros(d)  # Update term for theta
        u_t_theta_0 = 0  # Update term for theta_0
        p_previous = p_init  # Previous price iterate
        p_current = p_init  # Current price iterate
        p_next = 0  # Next price iterate
        theta_previous = theta_init  # Previous theta iterate
        theta_current = theta_init  # Current theta iterate
        theta_next = np.zeros(d)  # Next theta iterate
        theta_0_previous = theta_0_init  # Previous theta_0 iterate
        theta_0_current = theta_0_init  # Current theta_0 iterate
        theta_0_next = 0  # Next theta_0 iterate
        p_bar_cur = p_init  # Current average of price iterates
        p_bar_next = 0  # Next average of price iterates
        theta_bar_cur = theta_init  # Current average of theta iterates
        theta_bar_next = np.zeros(d)  # Next average of theta iterates
        theta_0_bar_cur = theta_0_init  # Current average of theta_0 iterates
        theta_0_bar_next = 0  # Next average of theta_0 iterates
        optim_gap = np.zeros(int(K / tresh_eval))  # Array to store optimality gaps
        feasib_gap = np.zeros(int(K / tresh_eval))  # Array to store feasibility gaps
        iteration_time_opconex = np.zeros(int(K / tresh_eval))  # Array to store iteration times
        lamda_norm = np.zeros(int(K / tresh_eval))  # Array to store norms of Lagrange multipliers
        grad_g_theta = -s_tild  # Gradient of constraint w.r.t. theta
        grad_g_theta_0 = -p_tild  # Gradient of constraint w.r.t. theta_0
        
        # Main loop of the opconex algorithm
        for k in range(1, K):
            # Update step sizes dynamically based on iteration k
            eta = (L_g * B) + (8 * L + 8 * M_g * B / p_max) + (8 * B * np.sqrt(var_g_grad)) * np.sqrt(k) / p_max
            tau = (9 * max(M_g, np.sqrt(var_g_grad)) * p_max) / B + (8 / B * np.sqrt(k) * np.sqrt(var_g + var_g_grad))
            if k == 1:
                # Compute initial optimality and feasibility gaps
                optim_gap[k-1] = p_star * D(s, p_star, theta_bar_cur, theta_0_bar_cur, xi_test_avg) - p_bar_cur * D(s, p_bar_cur, theta_star, theta_0_star, xi_test_avg)
                feasib_gap[k-1] = np.linalg.norm(np.maximum(-D_con(s_tild, p_tild, theta_bar_cur, theta_0_bar_cur, demand), 0))
                tStart_opconex = tp.time()  # Start timing the iteration
            if k % tresh_eval == 0:
                # Periodically evaluate gaps and record time
                iteration_time_opconex[int(k / tresh_eval)] = tp.time() - tStart_opconex  # Record elapsed time
                optim_gap[int(k / tresh_eval)] = p_star * D(s, p_star, theta_bar_cur, theta_0_bar_cur, xi_test_avg) - p_bar_cur * D(s, p_bar_cur, theta_star, theta_0_star, xi_test_avg)
                feasib_gap[int(k / tresh_eval)] = np.linalg.norm(np.maximum(-D_con(s_tild, p_tild, theta_bar_cur, theta_0_bar_cur, demand), 0))
                lamda_norm[int(k / tresh_eval)] = np.linalg.norm(lam_old)  # Record norm of Lagrange multiplier
                print(f"Iteration {k} of opconex completed...")  # Print progress
                tStart_opconex = tp.time()  # Reset timer
            
            # Compute constraint violation and update Lagrange multiplier
            s_para = (1 + theta_para) * ell_current - theta_para * ell_previous  # Weighted combination of constraint violations
            lam_new = np.maximum(lam_old + s_para / tau, 0)  # Update Lagrange multiplier, ensure non-negativity
            lam_old = lam_new  # Store previous Lagrange multiplier
            # Compute update terms for p, theta, and theta_0
            u_t_p = (1 + theta_para) * p_operator(s, p_current, theta_current, theta_0_current, xi[k + 1]) - \
                    theta_para * p_operator(s, p_previous, theta_previous, theta_0_previous, xi[k])
            p_next = min(max(0, (p_current - u_t_p / eta)), p_max)  # Update price, project to [0, p_max]
            u_t_theta = (1 + theta_para) * theta_operator(s, p_current) - theta_para * theta_operator(s, p_previous) + lam_new @ grad_g_theta
            theta_next = np.minimum(np.maximum(l, (theta_current - u_t_theta / eta)), u)  # Update theta, project to [l, u]
            u_t_theta_0 = (1 + theta_para) * theta_0_operator(p_current) - theta_para * theta_0_operator(p_previous) + lam_new @ grad_g_theta_0
            theta_0_next = np.minimum(np.maximum(l_0, (theta_0_current - u_t_theta_0 / eta)), u_0)  # Update theta_0, project to [l_0, u_0]
            # Compute next constraint violation
            ell_next = -D_con(s_tild, p_tild, theta_next, theta_0_next, demand)
            
            # Update previous and current values
            ell_previous = ell_current
            ell_current = ell_next
            p_previous = p_current
            p_current = p_next
            theta_previous = theta_current
            theta_current = theta_next
            theta_0_previous = theta_0_current
            theta_0_current = theta_0_next
            
            # Measure convergence with running averages
            p_bar_next = (p_next - p_bar_cur) / k + p_bar_cur  # Update average of price
            theta_bar_next = (theta_next - theta_bar_cur) / k + theta_bar_cur  # Update average of theta
            theta_0_bar_next = (theta_0_next - theta_0_bar_cur) / k + theta_0_bar_cur  # Update average of theta_0
            p_bar_cur = p_bar_next
            theta_bar_cur = theta_bar_next
            theta_0_bar_cur = theta_0_bar_next
        # Return convergence and performance metrics
        return optim_gap, feasib_gap, iteration_time_opconex, lamda_norm

    # Popex method for optimization
    def popex(d, m, K, l, l_0, u, u_0, p_max, p_init, theta_init, theta_0_init, B, xi, L, L_g, M_g, var_obj, var_g, var_g_grad, tresh_eval, p_star, theta_star, theta_0_star, xi_test_avg, xi_test_square, s, s_tild, theta_tild, theta_tild_0, demand, p_tild):
        # Initialize parameters
        theta_para = 1  # Momentum parameter for updates
        eta = 9 * L + np.sqrt(2 * K) * (B * (3 * M_g + 4 * np.sqrt(var_g_grad) * (1 + np.sqrt(var_g)))) / p_max  # Step size for p, theta, and theta_0
        tau = 2 * np.sqrt(2 * K) * (2 * M_g + 5 * np.sqrt(var_g_grad) + np.sqrt(var_g)) * p_max / B  # Step size for Lagrange multiplier
        s_para = 0  # Scalar for constraint violation
        lam_old = 0  # Previous Lagrange multiplier
        lam_new = 0  # Current Lagrange multiplier
        u_t_p = 0  # Update term for price
        u_t_theta = np.zeros(d)  # Update term for theta
        u_t_theta_0 = 0  # Update term for theta_0
        p_previous = p_init  # Previous price iterate
        p_current = p_init  # Current price iterate
        p_next = 0  # Next price iterate
        theta_previous = theta_init  # Previous theta iterate
        theta_current = theta_init  # Current theta iterate
        theta_next = np.zeros(d)  # Next theta iterate
        theta_0_previous = theta_0_init  # Previous theta_0 iterate
        theta_0_current = theta_0_init  # Current theta_0 iterate
        theta_0_next = 0  # Next theta_0 iterate
        p_bar_cur = p_init  # Current average of price iterates
        p_bar_next = 0  # Next average of price iterates
        theta_bar_cur = theta_init  # Current average of theta iterates
        theta_bar_next = np.zeros(d)  # Next average of theta iterates
        theta_0_bar_cur = theta_0_init  # Current average of theta_0 iterates
        theta_0_bar_next = 0  # Next average of theta_0 iterates
        optim_gap = np.zeros(int(K / tresh_eval))  # Array for optimality gaps
        feasib_gap = np.zeros(int(K / tresh_eval))  # Array for feasibility gaps
        iteration_time_popex = np.zeros(int(K / tresh_eval))  # Array for iteration times
        lamda_norm = np.zeros(int(K / tresh_eval))  # Array for norms of Lagrange multipliers
        grad_g_theta = -s_tild  # Gradient of constraint w.r.t. theta
        grad_g_theta_0 = -p_tild  # Gradient of constraint w.r.t. theta_0
        
        # Main loop of the popex algorithm
        for k in range(1, K):
            if k == 1:
                # Compute initial optimality and feasibility gaps
                optim_gap[k-1] = p_star * D(s, p_star, theta_bar_cur, theta_0_bar_cur, xi_test_avg) - p_bar_cur * D(s, p_bar_cur, theta_star, theta_0_star, xi_test_avg)
                feasib_gap[k-1] = np.linalg.norm(np.maximum(-D_con(s_tild, p_tild, theta_bar_cur, theta_0_bar_cur, demand), 0))
                tStart_popex = tp.time()  # Start timing
            if k % tresh_eval == 0:
                # Periodically evaluate gaps and record time
                iteration_time_popex[int(k / tresh_eval)] = tp.time() - tStart_popex  # Record elapsed time
                optim_gap[int(k / tresh_eval)] = p_star * D(s, p_star, theta_bar_cur, theta_0_bar_cur, xi_test_avg) - p_bar_cur * D(s, p_bar_cur, theta_star, theta_0_star, xi_test_avg)
                feasib_gap[int(k / tresh_eval)] = np.linalg.norm(np.maximum(-D_con(s_tild, p_tild, theta_bar_cur, theta_0_bar_cur, demand), 0))
                lamda_norm[int(k / tresh_eval)] = np.linalg.norm(lam_old)  # Record norm of Lagrange multiplier
                print(f"Iteration {k} of popex completed...")  # Print progress
                tStart_popex = tp.time()  # Reset timer
            
            # Compute constraint violation and update Lagrange multiplier
            s_para = -D_con(s_tild, p_tild, theta_current, theta_0_current, demand)
            lam_new = np.maximum(lam_old + s_para / tau, 0)  # Update Lagrange multiplier, ensure non-negativity
            lam_old = lam_new  # Store previous Lagrange multiplier
            # Compute update terms for p, theta, and theta_0
            u_t_p = (1 + theta_para) * p_operator(s, p_current, theta_current, theta_0_current, xi[k + 1]) - \
                    theta_para * p_operator(s, p_previous, theta_previous, theta_0_previous, xi[k])
            p_next = min(max(0, (p_current - u_t_p / eta)), p_max)  # Update price, project to [0, p_max]
            u_t_theta = (1 + theta_para) * theta_operator(s, p_current) - theta_para * theta_operator(s, p_previous) + lam_new @ grad_g_theta
            theta_next = np.minimum(np.maximum(l, (theta_current - u_t_theta / eta)), u)  # Update theta, project to [l, u]
            u_t_theta_0 = (1 + theta_para) * theta_0_operator(p_current) - theta_para * theta_0_operator(p_previous) + lam_new @ grad_g_theta_0
            theta_0_next = np.minimum(np.maximum(l_0, (theta_0_current - u_t_theta_0 / eta)), u_0)  # Update theta_0, project to [l_0, u_0]
            
            # Update previous and current iterates
            p_previous = p_current
            p_current = p_next
            theta_previous = theta_current
            theta_current = theta_next
            theta_0_previous = theta_0_current
            theta_0_current = theta_0_next
            
            # Measure convergence with running averages
            p_bar_next = (p_next - p_bar_cur) / k + p_bar_cur  # Update average of price
            theta_bar_next = (theta_next - theta_bar_cur) / k + theta_bar_cur  # Update average of theta
            theta_0_bar_next = (theta_0_next - theta_0_bar_cur) / k + theta_0_bar_cur  # Update average of theta_0
            p_bar_cur = p_bar_next
            theta_bar_cur = theta_bar_next
            theta_0_bar_cur = theta_0_bar_next
        # Return convergence and performance metrics
        return optim_gap, feasib_gap, iteration_time_popex, lamda_norm
    
    # RLSA method (from Alizadeh's Paper)
    def RLSA(d, m, K, l, l_0, u, u_0, p_max, p_init, theta_init, theta_0_init, B, xi, L, L_g, M_g, var_obj, var_g, var_g_grad, tresh_eval, p_star, theta_star, theta_0_star, xi_test_avg, xi_test_square, s, s_tild, theta_tild, theta_tild_0, demand, p_tild):
        # Initialize parameters
        C_f = np.linalg.norm(s_tild) + np.linalg.norm(p_tild)  # Constant based on norms of s_tild and p_tild
        gamma = np.sqrt(np.sqrt(m) / (12 * C_f))  # Initial step size for updates
        rho = np.sqrt(np.sqrt(m) / (12 * C_f))  # Initial step size for Lagrange multiplier
        t_0 = 1  # Initial weighting factor
        s_para = 0  # Scalar for constraint violation
        lam_old = 0  # Previous Lagrange multiplier
        lam_new = 0  # Current Lagrange multiplier
        rho_old = rho  # Previous step size for Lagrange multiplier
        t_old = t_0  # Previous weighting factor
        t_new = 0  # Next weighting factor
        rho_new = rho_old  # Next step size for Lagrange multiplier
        gamma_old = gamma  # Previous step size for updates
        gamma_new = 0  # Next step size for updates
        u_t_p = 0  # Update term for price
        u_t_theta = np.zeros(d)  # Update term for theta
        u_t_theta_0 = 0  # Update term for theta_0
        p_current = p_init  # Current price iterate
        p_next = 0  # Next price iterate
        theta_current = theta_init  # Current theta iterate
        theta_next = np.zeros(d)  # Next theta iterate
        theta_0_current = theta_0_init  # Current theta_0 iterate
        theta_0_next = 0  # Next theta_0 iterate
        sum_t = t_old  # Sum of weighting factors
        sum_t_p = t_old * p_current  # Running sum of weighted price iterates
        sum_t_theta = t_old * theta_current  # Running sum of weighted theta iterates
        sum_t_theta_0 = t_old * theta_0_current  # Running sum of weighted theta_0 iterates
        # Initial average solutions
        p_bar_cur = sum_t_p / sum_t  # Current weighted average of price
        theta_bar_cur = sum_t_theta / sum_t  # Current weighted average of theta
        theta_0_bar_cur = sum_t_theta_0 / sum_t  # Current weighted average of theta_0
        p_bar_next = 0  # Next weighted average of price
        theta_bar_next = np.zeros(d)  # Next weighted average of theta
        theta_0_bar_next = 0  # Next weighted average of theta_0
        optim_gap = np.zeros(int(K / tresh_eval))  # Array for optimality gaps
        feasib_gap = np.zeros(int(K / tresh_eval))  # Array for feasibility gaps
        iteration_time_RLSA = np.zeros(int(K / tresh_eval))  # Array for iteration times
        lamda_norm = np.zeros(int(K / tresh_eval))  # Array for norms of Lagrange multipliers
        grad_g_theta = -s_tild  # Gradient of constraint w.r.t. theta
        grad_g_theta_0 = -p_tild  # Gradient of constraint w.r.t. theta_0
        
        # Main loop of the RLSA algorithm
        for k in range(1, K):
            if k == 1:
                # Compute initial optimality and feasibility gaps
                optim_gap[k-1] = p_star * D(s, p_star, theta_bar_cur, theta_0_bar_cur, xi_test_avg) - p_bar_cur * D(s, p_bar_cur, theta_star, theta_0_star, xi_test_avg)
                feasib_gap[k-1] = np.linalg.norm(np.maximum(-D_con(s_tild, p_tild, theta_bar_cur, theta_0_bar_cur, demand), 0))
                tStart_RLSA = tp.time()  # Start timing
            if k % tresh_eval == 0:
                # Periodically evaluate gaps and record time
                iteration_time_RLSA[int(k / tresh_eval)] = tp.time() - tStart_RLSA  # Record elapsed time
                optim_gap[int(k / tresh_eval)] = p_star * D(s, p_star, theta_bar_cur, theta_0_bar_cur, xi_test_avg) - p_bar_cur * D(s, p_bar_cur, theta_star, theta_0_star, xi_test_avg)
                feasib_gap[int(k / tresh_eval)] = np.linalg.norm(np.maximum(-D_con(s_tild, p_tild, theta_bar_cur, theta_0_bar_cur, demand), 0))
                lamda_norm[int(k / tresh_eval)] = np.linalg.norm(lam_old)  # Record norm of Lagrange multiplier
                print(f"Iteration {k} of RLSA completed...")  # Print progress
                tStart_RLSA = tp.time()  # Reset timer
            
            # Compute constraint violation and update Lagrange multiplier
            s_para = -D_con(s_tild, p_tild, theta_current, theta_0_current, demand)
            lam_new = np.maximum(lam_old + s_para * rho_old, 0)  # Update Lagrange multiplier
            lam_old = lam_new  # Store previous Lagrange multiplier
            # Compute update terms for p, theta, and theta_0
            u_t_p = p_operator(s, p_current, theta_current, theta_0_current, xi[k + 1])
            p_next = min(max(0, (p_current - u_t_p * gamma_old)), p_max)  # Update price, project to [0, p_max]
            u_t_theta = theta_operator(s, p_current) + lam_new @ grad_g_theta
            theta_next = np.minimum(np.maximum(l, (theta_current - u_t_theta * gamma_old)), u)  # Update theta, project to [l, u]
            u_t_theta_0 = theta_0_operator(p_current) + lam_new @ grad_g_theta_0
            theta_0_next = np.minimum(np.maximum(l_0, (theta_0_current - u_t_theta_0 * gamma_old)), u_0)  # Update theta_0, project to [l_0, u_0]
            
            # Update current iterates
            p_current = p_next
            theta_current = theta_next
            theta_0_current = theta_0_next
            
            # Update step sizes and weighting factor dynamically
            rho_new = rho / (np.sqrt(k + 1) * np.log(k + 1))  # Adjust step size for Lagrange multiplier
            rho_old = rho_new  # Store new step size
            gamma_new = gamma / (np.sqrt(k + 1) * np.log(k + 1))  # Adjust step size for updates
            gamma_old = gamma_new  # Store new step size
            t_new = 1 / (np.sqrt(k + 1) * np.log(k + 1))  # Adjust weighting factor
            t_old = t_new  # Store new weighting factor
            
            # Update weighted sums for averages
            sum_t_p += t_new * p_next
            sum_t_theta += t_new * theta_next
            sum_t_theta_0 += t_new * theta_0_next
            sum_t += t_new
            
            # Measure convergence with weighted averages
            p_bar_next = sum_t_p / sum_t  # Compute weighted average of price
            theta_bar_next = sum_t_theta / sum_t  # Compute weighted average of theta
            theta_0_bar_next = sum_t_theta_0 / sum_t  # Compute weighted average of theta_0
            p_bar_cur = p_bar_next
            theta_bar_cur = theta_bar_next
            theta_0_bar_cur = theta_0_bar_next
        # Return convergence and performance metrics
        return optim_gap, feasib_gap, iteration_time_RLSA, lamda_norm
