from casadi import *
import numpy as np
import time
import os
import datetime
import calendar
import pandas
import ast


def solve_1node_mv(solver, solver_def, T0_f, q_array, c_array, init_bool, prev_sol):
    '''
    Solve instance of 1node MPC problem
    '''

    #initial condition
    T0 = (T0_f - 32.0) * (5.0 / 9.0) + 273.15

    #all parameters that changed for this instance of the problem
    par_init = [T0] + list(q_array) + list(c_array)

    # Solve the NLP
    if init_bool == False: #if it hasn't been solved before
        sol = solver(x0=solver_def['w0'], lbx=solver_def['lbw'], ubx=solver_def['ubw'], lbg=solver_def['lbg'],
                     ubg=solver_def['ubg'], p=par_init)
    elif init_bool == True: #if it has been solved before, use last solution to warm start
        sol = solver(x0=prev_sol['w0'], lbx=solver_def['lbw'], ubx=solver_def['ubw'], lbg=solver_def['lbg'],
                     ubg=solver_def['ubg'], p=par_init, lam_x0=prev_sol['lam_x0'], lam_g0=prev_sol['lam_g0'])

    #save solution to warm start next problem
    next_sol = {}
    next_sol['w0'] = sol['x'].full()
    next_sol['lam_x0'] = sol['lam_x'].full()
    next_sol['lam_g0'] = sol['lam_g'].full()

    #parse out important outputs
    x_full_opt = np.ndarray.flatten(sol['x'].full())
    x_full_opt2 = x_full_opt[1:]
    u_opt = x_full_opt2[0::2]
    x_opt = np.concatenate((np.array([x_full_opt[0]]), x_full_opt2[1::2]))
    x_opt_f = (x_opt - 273.15) * (9.0 / 5.0) + 32.0

    #number of iterations
    it_count = int(solver.stats()['iter_count'])
    print('Iterations: ' + str(solver.stats()['iter_count']))

    return u_opt, x_opt_f, next_sol, it_count


def setup_1node_mv(M, N, T, f_model_params, model_dir):
    '''
    setup 1node MPC problem
    '''

    #controller parameters
    os.chdir(model_dir)
    f = open(f_model_params, 'r')
    model_param_dict = f.read()
    f.close()
    model_param_dict = ast.literal_eval(model_param_dict)

    x_ref_low_f = model_param_dict['x_ref_low_f'] #soft lower temperature bound [F]
    x_ref_high_f = model_param_dict['x_ref_high_f'] #hard upper temperature bound [F]
    x_ref_f=x_ref_low_f
    u_max = model_param_dict['u_max'] #max power [W]
    u_min = model_param_dict['u_min'] #min power [W]
    T_in_f = model_param_dict['T_in_f'] #inlet temperature [F]
    T_amb_f = model_param_dict['T_amb_f'] #ambient temperature [F]
    vol_1node = model_param_dict['vol_1node'] # volume of 1node model
    U_1node = model_param_dict['U_1node'] #U valve for 1node model
    rho = model_param_dict['rho']  # [kg/m^3] #density of water
    cp = model_param_dict['cp']  # [J/kg*K] #heat capacity of water
    integrator = model_param_dict['integrator'] #numerical integrator (euler or RK4)

    x_ref_low = (x_ref_low_f - 32.0) * (5.0 / 9.0) + 273.15
    x_ref_high = (x_ref_high_f - 32.0) * (5.0 / 9.0) + 273.15
    x_ref=(x_ref_f-32.0)*(5.0/9.0)+273.15

    DT = float(T) / N / M

    # Declare model variables
    x = SX.sym('x')  # water temperature
    u = SX.sym('u')  # power
    draw_power = SX.sym('draw_power')  # draw power
    c = SX.sym('c')  # cost of electricity

    cap = vol_1node * cp * rho #thermal heat capacity

    T_amb = (T_amb_f - 32.0) * (5.0 / 9.0) + 273.15
    T_in = (T_in_f - 32.0) * (5.0 / 9.0) + 273.15

    # Model equations
    xdot=(u/(cap))+(U_1node/(cap))*(T_amb-x)-draw_power/cap


    # Objective term (electricity cost + thermal discomfort)
    L = c * (u) / 1000.0 + 0.01*fmax(0, x_ref_low - x) ** 2

    # Formulate discrete time dynamics
    f = Function('f', [x, u, draw_power, c], [xdot, L])
    X0 = SX.sym('X0', 1)
    U = SX.sym('U', 1)
    DRAW_POWER = SX.sym('DRAW_POWER')
    C = SX.sym('C')
    X = X0  # initialize state before integrating
    Q = 0
    # Fixed step Runge-Kutta 4 integrator
    if integrator == 'RK4':
        for j in range(M):
            k1, k1_q = f(X, U, DRAW_POWER, C)
            k2, k2_q = f(X + DT / 2 * k1, U, DRAW_POWER, C)
            k3, k3_q = f(X + DT / 2 * k2, U, DRAW_POWER, C)
            k4, k4_q = f(X + DT * k3, U, DRAW_POWER, C)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    #Euler integrator
    if integrator == 'euler':
        for j in range(M):
            k1, k1_q = f(X, U, DRAW_POWER, C)
            X = X + (DT) * k1
            Q = Q + (DT) * k1_q

    #one-step function
    F = Function('F', [X0, U, DRAW_POWER, C], [X, Q], ['x0', 'u', 'draw_power', 'c'], ['xf', 'qf'])

    #build NLP
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []
    par1 = []  # parameters for draw_power
    par2 = []  # parameters for cost
    par_t = []  # parameters for initial temperature

    Xk = SX.sym('Xk')
    T0x = SX.sym('T0x', 1)
    par_t = [T0x]

    w += [Xk]
    lbw += [280]
    ubw += [355]
    w0 += [310]

    g += [Xk - T0x]
    lbg += [0]
    ubg += [0]

    for k in range(N):
        # New NLP variable for the control
        Uk = SX.sym('U_' + str(k))
        w += [Uk]
        lbw += [u_min]
        ubw += [u_max]
        w0 += [(u_max - u_min) / 2]

        # Integrate till the end of the interval
        pk1 = SX.sym('Pk1_' + str(k), 1)
        pk2 = SX.sym('Pk2_' + str(k), 1)
        Fk = F(x0=Xk, u=Uk, draw_power=pk1, c=pk2)
        Xk_end = Fk['xf']
        J = J + Fk['qf']
        par1 += [pk1]
        par2 += [pk2]

        # new NLP variable for state at end of interval
        Xk = SX.sym('X_' + str(k + 1))
        w += [Xk]
        lbw += [280]
        ubw += [355]
        w0 += [310]

        # add equality constraint
        g += [Xk_end - Xk]
        lbg += [0]
        ubg += [0]

        # add max temp constraint
        g += [Xk]
        lbg += [0]
        ubg += [x_ref_high]

    par = par_t + par1 + par2

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(*par)}
    opts = {}
    opts['ipopt.print_level'] = 0
    solver = nlpsol('solver', 'ipopt', prob, opts);

    solver_def = {}
    solver_def['w0'] = w0
    solver_def['lbw'] = lbw
    solver_def['ubw'] = ubw
    solver_def['lbg'] = lbg
    solver_def['ubg'] = ubg

    return solver, solver_def





def solve_3node_mv(solver, solver_def, T0_l_f, T0_m_f, T0_h_f, q_array, c_array, init_bool, prev_sol):
    '''
    solve instance of 3node mpc problem
    '''

    #initial condition
    T0_l = (T0_l_f - 32.0) * (5.0 / 9.0) + 273.15
    T0_m = (T0_m_f - 32.0) * (5.0 / 9.0) + 273.15
    T0_h = (T0_h_f - 32.0) * (5.0 / 9.0) + 273.15

    par_init = [T0_l, T0_m, T0_h] + list(q_array) + list(c_array)

    # Solve the NLP
    if init_bool == False: #if this is the first time problem is being solved
        sol = solver(x0=solver_def['w0'], lbx=solver_def['lbw'], ubx=solver_def['ubw'], lbg=solver_def['lbg'],
                     ubg=solver_def['ubg'], p=par_init)
    elif init_bool == True: #if it has been solved before, warm-start with last solution
        sol = solver(x0=prev_sol['w0'], lbx=solver_def['lbw'], ubx=solver_def['ubw'], lbg=solver_def['lbg'],
                     ubg=solver_def['ubg'], p=par_init, lam_x0=prev_sol['lam_x0'], lam_g0=prev_sol['lam_g0'])

    #keep track of solution for next warm-start
    next_sol = {}
    next_sol['w0'] = sol['x'].full()
    next_sol['lam_x0'] = sol['lam_x'].full()
    next_sol['lam_g0'] = sol['lam_g'].full()

    #parse out relevant solution values
    x_full_opt = np.ndarray.flatten(sol['x'].full())
    x_full_opt2 = x_full_opt[3:]

    ul_opt = x_full_opt2[0::5]
    uh_opt = x_full_opt2[1::5]

    xl_opt = np.concatenate((np.array([x_full_opt[0]]), x_full_opt2[2::5]))
    xm_opt = np.concatenate((np.array([x_full_opt[1]]), x_full_opt2[3::5]))
    xh_opt = np.concatenate((np.array([x_full_opt[2]]), x_full_opt2[4::5]))

    xl_opt_f = (xl_opt - 273.15) * (9.0 / 5.0) + 32.0
    xm_opt_f = (xm_opt - 273.15) * (9.0 / 5.0) + 32.0
    xh_opt_f = (xh_opt - 273.15) * (9.0 / 5.0) + 32.0

    #solver iterations
    it_count = int(solver.stats()['iter_count'])
    print('Iterations: ' + str(solver.stats()['iter_count']))

    return ul_opt, uh_opt, xl_opt_f, xm_opt_f, xh_opt_f, next_sol, it_count


def setup_3node_mv(M, N, T, f_model_params, model_dir):
    '''
    setup 3node mpc problem
    '''

    #controller parameters
    os.chdir(model_dir)
    f = open(f_model_params, 'r')
    model_param_dict = f.read()
    f.close()
    model_param_dict = ast.literal_eval(model_param_dict)

    x_ref_low_f = model_param_dict['x_ref_low_f'] #soft lower bound on temperature
    x_ref_high_f = model_param_dict['x_ref_high_f'] #hard upper bound on temperature
    x_ref_f=x_ref_low_f
    u_max_l = model_param_dict['u_max_l'] #max power lower element [W]
    u_max_h = model_param_dict['u_max_h'] #max power upper element [W]
    u_min = model_param_dict['u_min'] #min power [W]
    T_in_f = model_param_dict['T_in_f'] #inlet temperature [F]
    T_amb_f = model_param_dict['T_amb_f'] #ambient temperature [F]
    rho = model_param_dict['rho'] #density of water
    cp = model_param_dict['cp'] #thermal heat capacity
    vol_l_3node = model_param_dict['vol_l_3node'] #3node lower volume
    vol_m_3node = model_param_dict['vol_m_3node'] #3node middle volume
    vol_h_3node = model_param_dict['vol_h_3node'] #3node upper volume
    U_l_3node = model_param_dict['U_l_3node'] #U value lower volume
    U_m_3node = model_param_dict['U_m_3node'] #U value middle volume
    U_h_3node = model_param_dict['U_h_3node'] #U value upper volume
    D_lm = model_param_dict['D_lm'] #thermal conductivity coefficent
    D_mh = model_param_dict['D_mh'] #thermal conductivity coefficient
    integrator = model_param_dict['integrator'] #numerical integrator

    x_ref_low = (x_ref_low_f - 32.0) * (5.0 / 9.0) + 273.15
    x_ref_high = (x_ref_high_f - 32.0) * (5.0 / 9.0) + 273.15
    x_ref=(x_ref_f-32.0)*(5.0/9.0)+273.15


    DT = float(T) / N / M

    # Declare model variables
    x_l = SX.sym('x_l')  # water temperature lower node
    x_m = SX.sym('x_m')  # water temperature middle node
    x_h = SX.sym('x_h')  # water temperature upper node

    u_l = SX.sym('u_l')  # power lower element
    u_h = SX.sym('u_h')  # power upper element

    draw_power = SX.sym('draw_power')  # draw power
    c = SX.sym('c')  # cost of electricity

    x = vertcat(x_l, x_m, x_h)
    u = vertcat(u_l, u_h)

    cap_l = vol_l_3node * cp * rho #thermal capacitance lower node
    cap_m = vol_m_3node * cp * rho #thermal capacitance middle node
    cap_h = vol_h_3node * cp * rho #thermal capacitance upper node

    T_amb = (T_amb_f - 32.0) * (5.0 / 9.0) + 273.15 #ambient temperature
    T_in = (T_in_f - 32.0) * (5.0 / 9.0) + 273.15 #inlet temperature

    #system dynamics
    xdot_l = (U_l_3node / (cap_l)) * (T_amb - x_l) - (1/(x_h-T_in))*(draw_power / cap_l) * (x_l - T_in) + (D_lm / cap_l) * (x_m - x_l)
    xdot_m = (u_l / (cap_m)) + (U_m_3node / (cap_m)) * (T_amb - x_m) - (1/(x_h-T_in))*(draw_power / cap_m) * (x_m - x_l) + (
                D_lm / cap_m) * (x_l - x_m) + (D_mh / cap_m) * (x_h - x_m)
    xdot_h = (u_h / (cap_h)) + (U_h_3node / (cap_h)) * (T_amb - x_h) - (1/(x_h-T_in))*(draw_power / cap_h) * (x_h - x_m) + (
                D_mh / cap_h) * (x_m - x_h)

    xdot = vertcat(xdot_l, xdot_m, xdot_h)

    # Objective term (electricity cost + thermal discomfort)
    L = c * (u_l + u_h) / 1000.0 + 0.01*fmax(0, x_ref_low - x_h) ** 2

    # Formulate discrete time dynamics
    f = Function('f', [x, u, draw_power, c], [xdot, L])
    X0 = SX.sym('X0', 3)
    U = SX.sym('U', 2)
    DRAW_POWER = SX.sym('DRAW_POWER')
    C = SX.sym('C')
    X = X0  # initialize state before integrating
    Q = 0
    # Fixed step Runge-Kutta 4 integrator
    if integrator == 'RK4':
        for j in range(M):
            k1, k1_q = f(X, U, DRAW_POWER, C)
            k2, k2_q = f(X + DT / 2 * k1, U, DRAW_POWER, C)
            k3, k3_q = f(X + DT / 2 * k2, U, DRAW_POWER, C)
            k4, k4_q = f(X + DT * k3, U, DRAW_POWER, C)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    # Fixed step euler integrator
    if integrator == 'euler':
        for j in range(M):
            k1, k1_q = f(X, U, DRAW_POWER, C)
            X = X + (DT) * k1
            Q = Q + (DT) * k1_q

    #one-step function
    F = Function('F', [X0, U, DRAW_POWER, C], [X, Q], ['x0', 'u', 'draw_power', 'c'], ['xf', 'qf'])

    #Build NLP
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []
    par1 = []  # parameters for draw power
    par2 = []  # parameters for cost
    par_t = []  # parameters for initial temperature

    Xk = SX.sym('Xk', 3)

    T0x = SX.sym('T0x', 3)
    par_t += [T0x]

    w += [Xk]
    lbw += [280, 280, 280]
    ubw += [355, 355, 355]
    w0 += [310, 310, 310]

    #initial condition
    g += [Xk - T0x]
    lbg += [0, 0, 0]
    ubg += [0, 0, 0]

    for k in range(N):
        # New NLP variable for the control
        Uk = SX.sym('U_' + str(k), 2)
        w += [Uk]
        lbw += [u_min, u_min]
        ubw += [u_max_l, u_max_h]
        w0 += [(u_max_l - u_min) / 2, (u_max_h - u_min) / 2]

        # Integrate till the end of the interval
        pk1 = SX.sym('Pk1_' + str(k), 1)
        pk2 = SX.sym('Pk2_' + str(k), 1)

        Fk = F(x0=Xk, u=Uk, draw_power=pk1, c=pk2)
        Xk_end = Fk['xf']
        J = J + Fk['qf']
        par1 += [pk1]
        par2 += [pk2]

        # new NLP variable for state at end of interval
        Xk = SX.sym('X_' + str(k + 1), 3)
        w += [Xk]
        lbw += [280, 280, 280]
        ubw += [355, 355, 355]
        w0 += [310, 310, 310]

        # add equality constraint
        g += [Xk_end - Xk]
        lbg += [0, 0, 0]
        ubg += [0, 0, 0]

        # Add inequality constraint
        g += [Xk[1] - Xk[0], Xk[2] - Xk[1]]
        lbg += [0, 0]
        ubg += [inf, inf]

        # Add non-conincient constraint
        g += [Uk[0] + Uk[1]]
        lbg += [0]
        ubg += [u_max_h]

        # Add max temp constraint
        g += [Xk[2]]
        lbg += [0]
        ubg += [x_ref_high]

    par = par_t + par1 + par2

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(*par)}
    opts = {}
    opts['ipopt.print_level'] = 0
    solver = nlpsol('solver', 'ipopt', prob, opts);

    solver_def = {}
    solver_def['w0'] = w0
    solver_def['lbw'] = lbw
    solver_def['ubw'] = ubw
    solver_def['lbg'] = lbg
    solver_def['ubg'] = ubg

    return solver, solver_def

