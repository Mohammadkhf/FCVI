# -*- coding: utf-8 -*-
"""
FCVI methods of three main algorithms for two different sets 
"""
import numpy as np
import time as tp
import cvxpy as cp

class algs:
    #%% opconex method
    def opconex(n, m, K, Q, c, D_x, D_y, c_xy, x_init, y_init, xi, L, L_g, M_g, beta_hat, setting, V, tresh_eval, x_star, y_star, x_til, xi_test_avg, xi_test_square):
        # Initialize parameters for the opconex algorithm
        theta = 1  # Momentum parameter for updates
        B = 5  # Constant used in step size calculations
        var_g_grad = 0  # Variance of gradient, initialized to zero
        var_g = 0  # Variance of constraint function, initialized to zero
        
        # Compute variance terms for setting 'set2'
        if setting == 'set2':
            var_g_grad = 4 * np.sum(np.einsum('ij,ij->j', V.T, V.T))  # Variance of gradient based on matrix V
            var_g = 2 * m + 4 * D_x**2 * np.sum(np.linalg.norm(V, axis=1)**2)  # Variance of constraint function
        
       # Compute step sizes eta and tau for optimization
        eta = (L_g * B) + (8 * L + 8 * M_g * B / D_x) + (8 * B * np.sqrt(var_g_grad)) * np.sqrt(K) / D_x  # Step size for x and y updates
        tau = (9 * max(M_g, np.sqrt(var_g_grad)) * D_x) / B + (8 / B * np.sqrt(K) * np.sqrt(var_g + var_g_grad))  # Step size for lambda updates
        
        # Initialize variables for 'set1' setting
        if setting == 'set1':
            s = 0  # Scalar for constraint violation term
            lam_old = 0  # Previous Lagrange multiplier
            lam_new = 0  # Current Lagrange multiplier
            ell_previous = np.dot((x_init + y_init), (x_init + y_init).T) - pow(c_xy, 2)  # Previous constraint violation value
            ell_current = ell_previous  # Current constraint violation value
            ell_next = 0  # Next constraint violation value
            u_t_x = np.zeros(n)  # Gradient update term for x
            u_t_y = np.zeros(n)  # Gradient update term for y
            x_previous = x_init  # Previous x iterate
            x_current = x_init  # Current x iterate
            x_next = np.zeros(n)  # Next x iterate
            y_previous = y_init  # Previous y iterate
            y_current = y_init  # Current y iterate
            y_next = np.zeros(n)  # Next y iterate
            x_bar_cur = x_init  # Current average of x iterates
            x_bar_next = np.zeros(n)  # Next average of x iterates
            y_bar_cur = y_init  # Current average of y iterates
            y_bar_next = np.zeros(n)  # Next average of y iterates
            optim_gap = np.zeros(int(K / tresh_eval))  # Array to store optimality gap at evaluation points
            feasib_gap = np.zeros(int(K / tresh_eval))  # Array to store feasibility gap at evaluation points
            iteration_time_opconex = np.zeros(int(K / tresh_eval))  # Array to store iteration times
            lamda_norm = np.zeros(int(K / tresh_eval))  # Array to store norms of Lagrange multipliers
            
            # Main loop of the opconex algorithm for 'set1'
            for k in range(1, K):
                if k == 1:
                    # Define and solve optimization problem to compute initial gaps
                    x_sol = cp.Variable(n)
                    y_sol = cp.Variable(n)
                    objective = cp.Maximize(x_bar_cur @ y_sol - x_sol.T @ y_bar_cur.T - x_sol.T @ c - 0.5 * cp.quad_form(x_sol, Q))
                    constraints = [
                        cp.norm(x_sol, 2) <= D_x,  # Constraint: x norm bounded by D_x
                        cp.norm(y_sol, 2) <= D_y,  # Constraint: y norm bounded by D_y
                        cp.norm(x_sol + y_sol, 2)**2 <= pow(c_xy, 2)  # Constraint: combined norm squared bounded by c_xy^2
                    ]
                    problem = cp.Problem(objective, constraints)
                    problem.solve()
                    optim_gap[k-1] = problem.value + 0.5 * x_bar_cur @ Q @ x_bar_cur.T + x_bar_cur @ c  # Compute initial optimality gap
                    feasib_gap[k-1] = max(0, (x_bar_cur + y_bar_cur) @ (x_bar_cur + y_bar_cur).T - pow(c_xy, 2))  # Compute initial feasibility gap
                    tStart_opconex = tp.time()  # Start timing the iteration
                if k % tresh_eval == 0:
                    # Periodically evaluate gaps and time
                    iteration_time_opconex[int(k / tresh_eval)] = tp.time() - tStart_opconex  # Record time for this interval
                    x_sol = cp.Variable(n)
                    y_sol = cp.Variable(n)
                    objective = cp.Maximize(x_bar_cur @ y_sol - x_sol.T @ y_bar_cur.T - x_sol.T @ c - 0.5 * cp.quad_form(x_sol, Q))
                    constraints = [
                        cp.norm(x_sol, 2) <= D_x,
                        cp.norm(y_sol, 2) <= D_y,
                        cp.norm(x_sol + y_sol, 2)**2 <= pow(c_xy, 2)
                    ]
                    problem = cp.Problem(objective, constraints)
                    problem.solve()
                    optim_gap[int(k / tresh_eval)] = problem.value + 0.5 * x_bar_cur @ Q @ x_bar_cur.T + x_bar_cur @ c  # Update optimality gap
                    feasib_gap[int(k / tresh_eval)] = max(0, (x_bar_cur + y_bar_cur) @ (x_bar_cur + y_bar_cur).T - pow(c_xy, 2))  # Update feasibility gap
                    lamda_norm[int(k / tresh_eval)] = np.linalg.norm(lam_old)  # Record norm of Lagrange multiplier
                    print(f"Iteration {k} of opconex completed...")  # Progress update
                    tStart_opconex = tp.time()  # Reset timer for next interval
                
                # Compute constraint violation and update Lagrange multiplier
                s = (1 + theta) * ell_current - theta * ell_previous  # Weighted combination of constraint violations
                lam_new = np.maximum(lam_old + s / tau, 0)  # Update Lagrange multiplier, ensuring non-negativity
                lam_old = lam_new  # Store previous Lagrange multiplier
                # Compute gradient-like updates for x and y
                u_t_x = (1 + theta) * ((Q @ x_current) + c + y_current) + 2 * lam_new * (x_current + y_current) - \
                        theta * ((Q @ x_previous) + c + y_previous)
                u_t_y = -(1 + theta) * x_current + theta * x_previous + 2 * lam_new * (x_current + y_current)
                # Update x and y with projection to satisfy norm constraints
                x_next = (x_current - u_t_x / eta) * min(1, D_x / np.linalg.norm(x_current - u_t_x / eta))
                y_next = (y_current - u_t_y / eta) * min(1, D_y / np.linalg.norm(y_current - u_t_y / eta))
                # Compute next constraint violation value
                ell_next = np.dot((x_current + y_current), (x_current + y_current).T) - pow(c_xy, 2) + \
                           2 * (x_current + y_current) @ (x_next - x_current).T + 2 * (x_current + y_current) @ (y_next - y_current).T
                # Update previous and current values
                ell_previous = ell_current
                ell_current = ell_next
                x_previous = x_current
                x_current = x_next
                y_previous = y_current
                y_current = y_next
                # Update running averages for convergence measures
                x_bar_next = (x_next - x_bar_cur) / k + x_bar_cur
                y_bar_next = (y_next - y_bar_cur) / k + y_bar_cur
                x_bar_cur = x_bar_next
                y_bar_cur = y_bar_next

        # Handle 'set2' setting
        if setting == 'set2':
            s = np.zeros(m)  # Array for constraint violation terms
            eta = eta  # Step size for x and y updates
            tau = tau  # Step size for lambda updates
            lam_old = np.zeros(m)  # Previous Lagrange multipliers
            lam_new = np.zeros(m)  # Current Lagrange multipliers
            ell_previous = (np.sum((x_init[:, None] - x_til) * V.T, axis=0) + xi[0, :]) ** 2 - beta_hat  # Initial constraint violation
            ell_current = ell_previous  # Current constraint violation
            ell_next = np.zeros(m)  # Next constraint violation
            grad_g_x = np.zeros((n, m))  # Gradient of constraint function
            u_t_x = np.zeros(n)  # Gradient update term for x
            u_t_y = np.zeros(n)  # Gradient update term for y
            lamda_norm = np.zeros(int(K / tresh_eval))  # Array for norms of Lagrange multipliers
            x_previous = x_init  # Previous x iterate
            x_current = x_init  # Current x iterate
            x_next = np.zeros(n)  # Next x iterate
            y_previous = y_init  # Previous y iterate
            y_current = y_init  # Current y iterate
            y_next = np.zeros(n)  # Next y iterate
            x_bar_cur = x_init  # Current average of x iterates
            x_bar_next = np.zeros(n)  # Next average of x iterates
            y_bar_cur = y_init  # Current average of y iterates
            y_bar_next = np.zeros(n)  # Next average of y iterates
            optim_gap = np.zeros(int(K / tresh_eval))  # Array for optimality gaps
            feasib_gap = np.zeros(int(K / tresh_eval))  # Array for feasibility gaps
            iteration_time_opconex = np.zeros(int(K / tresh_eval))  # Array for iteration times
            
            # Main loop for 'set2'
            for k in range(1, K):
                if k == 1:
                    # Compute initial optimality and feasibility gaps
                    optim_gap[k-1] = 0.5 * x_bar_cur @ Q @ x_bar_cur.T + x_bar_cur @ c + x_bar_cur @ y_star - 0.5 * x_star.T @ Q @ x_star - c.T @ x_star - y_bar_cur @ x_star
                    feasib_gap[k-1] = np.linalg.norm(np.maximum(np.sum(((x_bar_cur[:, None] - x_til) * V.T)**2, axis=0) \
                        + 2 * np.sum((x_bar_cur[:, None] - x_til) * V.T, axis=0) * xi_test_avg + xi_test_square - beta_hat, 0))
                    tStart_opconex = tp.time()  # Start timing
                if k % tresh_eval == 0:
                    # Periodically compute gaps and time
                    iteration_time_opconex[int(k / tresh_eval)] = tp.time() - tStart_opconex  # Record time
                    optim_gap[int(k / tresh_eval)] = 0.5 * x_bar_cur @ Q @ x_bar_cur.T + x_bar_cur @ c + x_bar_cur @ y_star - 0.5 * x_star.T @ Q @ x_star - c.T @ x_star - y_bar_cur @ x_star
                    feasib_gap[int(k / tresh_eval)] = np.linalg.norm(np.maximum(np.sum(((x_bar_cur[:, None] - x_til) * V.T)**2, axis=0) \
                        + 2 * np.sum((x_bar_cur[:, None] - x_til) * V.T, axis=0) * xi_test_avg + xi_test_square - beta_hat, 0))
                    lamda_norm[int(k / tresh_eval)] = np.linalg.norm(lam_old)  # Record norm of Lagrange multiplier
                    tStart_opconex = tp.time()  # Reset timer
                    print(f"Iteration {k} of opconex completed...")  # Progress update
                # Compute constraint violation and update Lagrange multiplier
                s = (1 + theta) * ell_current - theta * ell_previous  # Weighted constraint violation
                lam_new = np.maximum(lam_old + s / tau, 0)  # Update Lagrange multiplier
                lam_old = lam_new  # Store previous value
                # Compute gradient of constraint function
                grad_g_x = 2 * np.multiply(np.sum((x_current[:, None] - x_til) * V.T, axis=0) + xi[K + k, :], V.T)
                # Compute update terms for x and y
                u_t_x = (1 + theta) * ((Q @ x_current) + c + y_current) - theta * ((Q @ x_previous) + c + y_previous) + lam_new @ grad_g_x.T
                u_t_y = -(1 + theta) * x_current + theta * x_previous
                # Update x and y with projection
                x_next = (x_current - u_t_x / eta) * min(1, D_x / np.linalg.norm(x_current - u_t_x / eta))
                y_next = (y_current - u_t_y / eta) * min(1, D_y / np.linalg.norm(y_current - u_t_y / eta))
                # Compute next constraint violation
                ell_next = (np.sum((x_current[:, None] - x_til) * V.T, axis=0) + xi[k, :]) ** 2 - beta_hat + \
                           2 * (np.sum((x_current[:, None] - x_til) * V.T, axis=0) + xi[k, :]) * (V @ (x_next - x_current))
                # Update previous and current values
                ell_previous = ell_current
                ell_current = ell_next
                x_previous = x_current
                x_current = x_next
                y_previous = y_current
                y_current = y_next
                # Update running averages for convergence
                x_bar_next = (x_next - x_bar_cur) / k + x_bar_cur
                y_bar_next = (y_next - y_bar_cur) / k + y_bar_cur
                x_bar_cur = x_bar_next
                y_bar_cur = y_bar_next
        # Return convergence and performance metrics
        return optim_gap, feasib_gap, iteration_time_opconex, lamda_norm

    #%% popex method
    def popex(n, m, K, Q, c, D_x, D_y, c_xy, x_init, y_init, xi, L, L_g, M_g, beta_hat, setting, V, tresh_eval, x_star, y_star, x_til, xi_test_avg, xi_test_square):
        # Initialize parameters for the popex algorithm
        theta = 1  # Momentum parameter
        B = 5  # Constant for step size calculations
        var_g_grad = 0  # Variance of gradient
        var_g = 0  # Variance of constraint function
        
        # Handle 'set1' setting
        if setting == 'set1':
            # Set step sizes eta and tau
            eta = 9 * L + np.sqrt(2 * K) * (B * (3 * M_g + 4 * np.sqrt(var_g_grad) * (1 + np.sqrt(var_g)))) / D_x
            tau = 2 * np.sqrt(2 * K) * (2 * M_g + 5 * np.sqrt(var_g_grad) + np.sqrt(var_g)) * D_x / B
            
            # Initialize variables
            s = 0  # Scalar constraint violation term
            lam_old = 0  # Previous Lagrange multiplier
            lam_new = 0  # Current Lagrange multiplier
            u_t_x = np.zeros(n)  # Gradient update term for x
            u_t_y = np.zeros(n)  # Gradient update term for y
            x_previous = x_init  # Previous x iterate
            x_current = x_init  # Current x iterate
            x_next = np.zeros(n)  # Next x iterate
            y_previous = y_init  # Previous y iterate
            y_current = y_init  # Current y iterate
            y_next = np.zeros(n)  # Next y iterate
            x_bar_cur = x_init  # Current average of x iterates
            x_bar_next = np.zeros(n)  # Next average of x iterates
            y_bar_cur = y_init  # Current average of y iterates
            y_bar_next = np.zeros(n)  # Next average of y iterates
            optim_gap = np.zeros(int(K / tresh_eval))  # Array for optimality gaps
            feasib_gap = np.zeros(int(K / tresh_eval))  # Array for feasibility gaps
            iteration_time_popex = np.zeros(int(K / tresh_eval))  # Array for iteration times
            lamda_norm = np.zeros(int(K / tresh_eval))  # Array for norms of Lagrange multipliers
            
            # Main loop for 'set1'
            for k in range(1, K):
                if k == 1:
                    # Compute initial optimality and feasibility gaps
                    x_sol = cp.Variable(n)
                    y_sol = cp.Variable(n)
                    objective = cp.Maximize(x_bar_cur.T @ y_sol - x_sol.T @ y_bar_cur - x_sol.T @ c - 0.5 * cp.quad_form(x_sol, Q))
                    constraints = [
                        cp.norm(x_sol, 2) <= D_x,
                        cp.norm(y_sol, 2) <= D_y,
                        cp.norm(x_sol + y_sol, 2)**2 <= pow(c_xy, 2)
                    ]
                    problem = cp.Problem(objective, constraints)
                    problem.solve()
                    optim_gap[k-1] = problem.value + 0.5 * x_bar_cur.T @ Q @ x_bar_cur + x_bar_cur.T @ c
                    feasib_gap[k-1] = max(0, (x_bar_cur + y_bar_cur).T @ (x_bar_cur + y_bar_cur) - pow(c_xy, 2))
                    tStart_popex = tp.time()  # Start timing
                if k % tresh_eval == 0:
                    # Periodically compute gaps and time
                    iteration_time_popex[int(k / tresh_eval)] = tp.time() - tStart_popex  # Record time
                    x_sol = cp.Variable(n)
                    y_sol = cp.Variable(n)
                    objective = cp.Maximize(x_bar_cur.T @ y_sol - x_sol.T @ y_bar_cur - x_sol.T @ c - 0.5 * cp.quad_form(x_sol, Q))
                    constraints = [
                        cp.norm(x_sol, 2) <= D_x,
                        cp.norm(y_sol, 2) <= D_y,
                        cp.norm(x_sol + y_sol, 2)**2 <= pow(c_xy, 2)
                    ]
                    problem = cp.Problem(objective, constraints)
                    problem.solve()
                    optim_gap[int(k / tresh_eval)] = problem.value + 0.5 * x_bar_cur.T @ Q @ x_bar_cur + x_bar_cur.T @ c
                    feasib_gap[int(k / tresh_eval)] = max(0, (x_bar_cur + y_bar_cur).T @ (x_bar_cur + y_bar_cur) - pow(c_xy, 2))
                    lamda_norm[int(k / tresh_eval)] = np.linalg.norm(lam_old)  # Record norm of Lagrange multiplier
                    print(f"Iteration {k} of popex completed...")  # Progress update
                    tStart_popex = tp.time()  # Reset timer
                # Compute constraint violation and update Lagrange multiplier
                s = np.dot((x_current + y_current).T, (x_current + y_current)) - pow(c_xy, 2)
                lam_new = np.maximum(lam_old + s / tau, 0)  # Update Lagrange multiplier
                lam_old = lam_new  # Store previous value
                # Compute update terms for x and y
                u_t_x = (1 + theta) * ((Q @ x_current) + c + y_current) + 2 * lam_new * (x_current + y_current) - \
                        theta * ((Q @ x_previous) + c + y_previous)
                u_t_y = -(1 + theta) * x_current + theta * x_current + 2 * lam_new * (x_current + y_current)
                # Update x and y with projection
                x_next = (x_current - u_t_x / eta) * min(1, D_x / np.linalg.norm(x_current - u_t_x / eta))
                y_next = (y_current - u_t_y / eta) * min(1, D_y / np.linalg.norm(y_current - u_t_y / eta))
                # Update previous and current iterates
                x_previous = x_current
                x_current = x_next
                y_previous = y_current
                y_current = y_next
                # Update running averages for convergence
                x_bar_next = (x_next - x_bar_cur) / k + x_bar_cur
                y_bar_next = (y_next - y_bar_cur) / k + y_bar_cur
                x_bar_cur = x_bar_next
                y_bar_cur = y_bar_next
        
        # Handle 'set2' setting
        if setting == 'set2':
            # Define number of epochs and iterations per epoch
            epochs = 5  # Number of restart epochs
            K_epoch = K / epochs  # Iterations per epoch
            
            # Compute variance terms for step sizes
            var_g_grad = 4 * np.sum(np.einsum('ij,ij->j', V.T, V.T))  # Variance of gradient
            var_g = 2 * m + 4 * D_x**2 * np.sum(np.linalg.norm(V, axis=1)**2)  # Variance of constraint
            # Set step sizes eta and tau
            eta = 9 * L + np.sqrt(2 * K_epoch) * (B * (3 * M_g + 4 * np.sqrt(var_g_grad) * (1 + np.sqrt(var_g)))) / D_x
            tau = 2 * np.sqrt(2 * K_epoch) * (2 * M_g + 5 * np.sqrt(var_g_grad) + np.sqrt(var_g)) * D_x / B
            
            # Adjust step sizes for stability
            eta = eta / 50
            tau = tau / 50
            
            # Initialize variables
            s = np.zeros(m)  # Constraint violation array
            lam_old = np.zeros(m)  # Previous Lagrange multipliers
            lam_new = lam_old  # Current Lagrange multipliers
            u_t_x = np.zeros(n)  # Gradient update term for x
            u_t_y = np.zeros(n)  # Gradient update term for y
            x_previous = x_init  # Previous x iterate
            x_current = x_init  # Current x iterate
            x_next = np.zeros(n)  # Next x iterate
            y_previous = y_init  # Previous y iterate
            y_current = y_init  # Current y iterate
            y_next = np.zeros(n)  # Next y iterate
            x_bar_cur = x_init  # Current average of x iterates
            x_bar_next = np.zeros(n)  # Next average of x iterates
            y_bar_cur = y_init  # Current average of y iterates
            y_bar_next = np.zeros(n)  # Next average of y iterates
            grad_g_x = np.zeros((n, m))  # Gradient of constraint function
            optim_gap = np.zeros(int(K / tresh_eval))  # Array for optimality gaps
            feasib_gap = np.zeros(int(K / tresh_eval))  # Array for feasibility gaps
            iteration_time_popex = np.zeros(int(K / tresh_eval))  # Array for iteration times
            lamda_norm = np.zeros(int(K / tresh_eval))  # Array for norms of Lagrange multipliers
            
            # Main loop for 'set2'
            for k in range(1, K):
                if k == 1:
                    # Compute initial optimality and feasibility gaps
                    optim_gap[k-1] = 0.5 * x_bar_cur @ Q @ x_bar_cur.T + x_bar_cur @ c + x_bar_cur @ y_star - 0.5 * x_star.T @ Q @ x_star - c.T @ x_star - y_bar_cur @ x_star
                    feasib_gap[k-1] = np.linalg.norm(np.maximum(np.sum(((x_bar_cur[:, None] - x_til) * V.T)**2, axis=0) \
                        + 2 * np.sum((x_bar_cur[:, None] - x_til) * V.T, axis=0) * xi_test_avg + xi_test_square - beta_hat, 0))
                    tStart_popex = tp.time()  # Start timing
                if k % tresh_eval == 0:
                    # Periodically compute gaps and time
                    iteration_time_popex[int(k / tresh_eval)] = tp.time() - tStart_popex  # Record time
                    optim_gap[int(k / tresh_eval)] = 0.5 * x_bar_cur @ Q @ x_bar_cur.T + x_bar_cur @ c + x_bar_cur @ y_star - 0.5 * x_star @ Q @ x_star.T - c.T @ x_star - y_bar_cur @ x_star
                    feasib_gap[int(k / tresh_eval)] = np.linalg.norm(np.maximum(np.sum(((x_bar_cur[:, None] - x_til) * V.T)**2, axis=0) \
                        + 2 * np.sum((x_bar_cur[:, None] - x_til) * V.T, axis=0) * xi_test_avg + xi_test_square - beta_hat, 0))
                    lamda_norm[int(k / tresh_eval)] = np.linalg.norm(lam_old)  # Record norm of Lagrange multiplier
                    print(f"Iteration {k} of popex completed...")  # Progress update
                    tStart_popex = tp.time()  # Reset timer
                if k % K_epoch == 0:
                    # Restart Lagrange multiplier at epoch boundary
                    lam_old = np.zeros(m)
                # Compute constraint violation and update Lagrange multiplier
                s = (np.sum((x_current[:, None] - x_til) * V.T, axis=0) + xi[k, :]) ** 2 - beta_hat
                lam_new = np.maximum(lam_old + s / tau, 0)  # Update Lagrange multiplier
                lam_old = lam_new  # Store previous value
                # Compute gradient of constraint function
                grad_g_x = 2 * np.multiply(np.sum((x_current[:, None] - x_til) * V.T, axis=0) + xi[K + k, :], V.T)
                # Compute update terms for x and y
                u_t_x = ((1 + theta) * ((Q @ x_current) + c.T + y_current) - theta * ((Q @ x_previous) + c.T + y_previous)) + lam_new @ grad_g_x.T
                u_t_y = (-(1 + theta) * x_current + theta * x_previous)
                # Update x and y with projection
                x_next = (x_current - u_t_x / eta) * min(1, D_x / np.linalg.norm(x_current - u_t_x / eta))
                y_next = (y_current - u_t_y / eta) * min(1, D_y / np.linalg.norm(y_current - u_t_y / eta))
                # Update previous and current iterates
                x_previous = x_current
                x_current = x_next
                y_previous = y_current
                y_current = y_next
                # Update running averages for convergence
                x_bar_next = (x_next - x_bar_cur) / k + x_bar_cur
                y_bar_next = (y_next - y_bar_cur) / k + y_bar_cur
                x_bar_cur = x_bar_next
                y_bar_cur = y_bar_next
        # Return convergence and performance metrics
        return optim_gap, feasib_gap, iteration_time_popex, lamda_norm

    #%% adopex method
    def adopex(n, K, Q, c, D_x, D_y, c_xy, x_init, y_init, L, L_g, M_g, tresh_eval):
        # Initialize parameters for the adopex algorithm
        c_1 = 4  # Constant for step size eta
        c_2 = 12  # Constant for adaptive step size adjustment
        beta = 12 * pow(M_g, 2) / (pow(c_1 * L, 2))  # Parameter for tau step size
        theta_old = 1  # Previous momentum parameter
        theta_new = 0  # Next momentum parameter
        eta_old = c_1 * L  # Initial step size for x and y updates
        eta_new = 0  # Next step size
        tau_old = beta * eta_old  # Initial step size for lambda updates
        tau_new = 0  # Next step size
        gamma_old = 1  # Previous weighting factor for averages
        gamma_new = 0  # Next weighting factor
        
        # Initialize iterates and variables
        s = 0  # Scalar constraint violation term
        lam_previous = 0  # Previous Lagrange multiplier
        lam_current = 0  # Current Lagrange multiplier
        lam_next = 0  # Next Lagrange multiplier
        u_t_x = np.zeros(n)  # Gradient update term for x
        u_t_y = np.zeros(n)  # Gradient update term for y
        x_previous = x_init  # Previous x iterate
        x_current = x_init  # Current x iterate
        x_next = np.zeros(n)  # Next x iterate
        y_previous = y_init  # Previous y iterate
        y_current = y_init  # Current y iterate
        y_next = np.zeros(n)  # Next y iterate
        lamda_norm = np.zeros(int(K / tresh_eval))  # Array for norms of Lagrange multipliers
        # Measures of convergence and performance
        optim_gap = np.zeros(int(K / tresh_eval))  # Array for optimality gaps
        feasib_gap = np.zeros(int(K / tresh_eval))  # Array for feasibility gaps
        iteration_time_adopex = np.zeros(int(K / tresh_eval))  # Array for iteration times
        # Weighted sum initialization for averages
        sum_gamma = gamma_old  # Sum of gamma weights
        sum_gamma_x = gamma_old * x_current  # Running sum of gamma-weighted x iterates
        sum_gamma_y = gamma_old * y_current  # Running sum of gamma-weighted y iterates
        # Initial average solutions
        x_bar_cur = sum_gamma_x / sum_gamma  # Current weighted average of x
        x_bar_next = np.zeros(n)  # Next weighted average of x
        y_bar_cur = sum_gamma_y / sum_gamma  # Current weighted average of y
        y_bar_next = np.zeros(n)  # Next weighted average of y
        
        # Main loop of the adopex algorithm
        for k in range(1, K):
            if k == 1:
                # Compute initial optimality and feasibility gaps
                x_sol = cp.Variable(n)
                y_sol = cp.Variable(n)
                objective = cp.Maximize(x_bar_cur.T @ y_sol - x_sol.T @ y_bar_cur - x_sol.T @ c - 0.5 * cp.quad_form(x_sol, Q))
                constraints = [
                    cp.norm(x_sol, 2) <= D_x,
                    cp.norm(y_sol, 2) <= D_y,
                    cp.norm(x_sol + y_sol, 2)**2 <= pow(c_xy, 2)
                ]
                problem = cp.Problem(objective, constraints)
                problem.solve()
                optim_gap[k-1] = problem.value + 0.5 * x_bar_cur.T @ Q @ x_bar_cur + x_bar_cur.T @ c
                feasib_gap[k-1] = max(0, (x_bar_cur + y_bar_cur).T @ (x_bar_cur + y_bar_cur) - pow(c_xy, 2))
                tStart_adopex = tp.time()  # Start timing
            if k % tresh_eval == 0:
                # Periodically compute gaps and time
                iteration_time_adopex[int(k / tresh_eval)] = tp.time() - tStart_adopex  # Record time
                x_sol = cp.Variable(n)
                y_sol = cp.Variable(n)
                objective = cp.Maximize(x_bar_cur.T @ y_sol - x_sol.T @ y_bar_cur - x_sol.T @ c - 0.5 * cp.quad_form(x_sol, Q))
                constraints = [
                    cp.norm(x_sol, 2) <= D_x,
                    cp.norm(y_sol, 2) <= D_y,
                    cp.norm(x_sol + y_sol, 2)**2 <= pow(c_xy, 2)
                ]
                problem = cp.Problem(objective, constraints)
                problem.solve()
                optim_gap[int(k / tresh_eval)] = problem.value + 0.5 * x_bar_cur.T @ Q @ x_bar_cur + x_bar_cur.T @ c
                feasib_gap[int(k / tresh_eval)] = max(0, (x_bar_cur + y_bar_cur).T @ (x_bar_cur + y_bar_cur) - pow(c_xy, 2))
                lamda_norm[int(k / tresh_eval)] = np.linalg.norm(lam_current)  # Record norm of Lagrange multiplier
                print(f"Iteration {k} of adopex completed...")  # Progress update
                tStart_adopex = tp.time()  # Reset timer
            # Compute constraint violation and update Lagrange multiplier
            s = (1 + theta_old) * (np.dot((x_current + y_current).T, (x_current + y_current)) - pow(c_xy, 2)) - \
                theta_old * (np.dot((x_previous + y_previous).T, (x_previous + y_previous)) - pow(c_xy, 2))
            lam_next = np.maximum(lam_current + s / tau_old, 0)  # Update Lagrange multiplier
            # Compute update terms for x and y
            u_t_x = (1 + theta_old) * ((Q @ x_current) + c + y_current + 2 * lam_current * (x_current + y_current)) - \
                    theta_old * ((Q @ x_previous) + c + y_previous + 2 * lam_previous * (x_previous + y_previous))
            u_t_y = (1 + theta_old) * (-x_current + 2 * lam_current * (x_current + y_current)) - \
                    theta_old * (-x_previous + 2 * lam_previous * (x_previous + y_previous))
            # Update x and y with projection
            x_next = (x_current - u_t_x / eta_old) * min(1, D_x / np.linalg.norm(x_current - u_t_x / eta_old))
            y_next = (y_current - u_t_y / eta_old) * min(1, D_y / np.linalg.norm(y_current - u_t_y / eta_old))
            # Update previous and current iterates
            x_previous = x_current
            x_current = x_next
            y_previous = y_current
            y_current = y_next
            # Update parameters adaptively
            eta_new = c_1 * L + c_2 * L_g * np.linalg.norm(lam_current)  # Adaptive step size for x and y
            lam_previous = lam_current  # Store previous Lagrange multiplier
            lam_current = lam_next  # Update current Lagrange multiplier
            eta_old = eta_new  # Update step size
            gamma_new = c_1 * L / eta_new  # Compute new weighting factor
            theta_new = gamma_old / gamma_new  # Update momentum parameter
            gamma_old = gamma_new  # Update weighting factor
            tau_new = beta * eta_new  # Update step size for lambda
            tau_old = tau_new  # Store new step size
            theta_old = theta_new  # Store new momentum value
            # Update weighted sums for averages
            sum_gamma_x += gamma_new * x_next
            sum_gamma_y += gamma_new * y_next
            sum_gamma += gamma_new
            # Compute weighted averages for convergence
            x_bar_next = sum_gamma_x / sum_gamma  # Weighted average of x
            y_bar_next = sum_gamma_y / sum_gamma  # Weighted average of y
            x_bar_cur = x_bar_next
            y_bar_cur = y_bar_next
        # Return convergence and performance metrics
        return optim_gap, feasib_gap, iteration_time_adopex, lamda_norm
