import numpy as np
#from casadi import *
import ast
import os

def sim_multinode_mpc_thermostat(wh_params, sim_dt, mpc_dt, q_array_target, T0,state_last,f_model_params,model_dir,sensor_ind):

    #Controller parameters
    os.chdir(model_dir)
    f=open(f_model_params,'r')
    model_param_dict=f.read()
    f.close()
    model_param_dict=ast.literal_eval(model_param_dict)

    x_ref_low_f=model_param_dict['x_ref_low_f'] #lower temperature bound [F]
    x_ref_high_f=model_param_dict['x_ref_high_f'] #upper temperature bound [F]
    if '3node' in f_model_params:
        u_max_l=model_param_dict['u_max_l'] #upper power bound for lower element
        u_max_h=model_param_dict['u_max_h'] #upper power bound for upper element
    elif '1node' in f_model_params:
        u_max_l=model_param_dict['u_max'] #upper power bound for lower element
        u_max_h=model_param_dict['u_max'] #upper power bound for upper element
    x_ref_low=(x_ref_low_f-32.0)*(5.0/9.0)+273.15 #lower temperature bound [K]
    x_ref_high=(x_ref_high_f-32.0)*(5.0/9.0)+273.15 #upper temperature bound [K]

    M = wh_params['M'] #number of nodes
    element_bot_ind = wh_params['element_bot_ind'] #node index of bottom element
    element_top_ind = wh_params['element_top_ind'] #node index of top element
    diff = wh_params['diff'] #diffusion coefficient
    A_surf = wh_params['A_surf'] #surface area
    slug_cross_area = wh_params['slug_cross_area'] #cross sectional area
    cap = wh_params['cap'] #thermal capacitance
    R = wh_params['R'] #Thermal resistiance of insulation
    T_amb = wh_params['T_amb']  #ambient temperature [K]
    dx = wh_params['dx'] #node height
    T_in = wh_params['T_in'] #inlet temperature [K]
    #mv_bool = wh_params['mv_bool']
    mv_set_f = wh_params['mv_set_f'] #mixing valve setpoint
    rho=wh_params['rho'] #water density
    cp=wh_params['cp'] # thermal heat capacity
    mv_set = (mv_set_f - 32) * (5.0 / 9.0) + 273.15

    N_sim = int(mpc_dt / sim_dt)
    temp_sim_k = np.zeros((N_sim + 1, M))
    temp_sim_k[0, :] = T0

    loss_coef=np.ones((M,))*(A_surf/R)
    loss_coef[0]=(A_surf+slug_cross_area*1.2)/R
    loss_coef[M-1]=(A_surf+slug_cross_area*1.7)/R

    amb_loss_array=[]
    flowrate_metric_h_array=[]
    q_array_actual_array=[]
    flowrate_metric_m_array=[]

    Q_l_array=np.zeros((N_sim,))
    Q_h_array=np.zeros((N_sim,))

    #loop through timesteps
    for i in range(N_sim):
        temp_now_k=temp_sim_k[i,sensor_ind]
        T0_m_k_thermostat = temp_now_k[6]
        T0_h_k_thermostat = temp_now_k[7]

        #Thermostatic control logic

        # if top isnt already on
        if (state_last == 1) or (state_last == 0):
            # turn on if need be
            if T0_h_k_thermostat <= x_ref_low:
                state = 2
            # if bottom is off
            elif state_last == 0:
                # see if it needs to be turned on
                if T0_m_k_thermostat <= x_ref_low:
                    state = 1
                else:
                    state = 0
            # if bottom is on
            elif state_last == 1:
                if T0_m_k_thermostat > x_ref_high:
                    state = 0
                else:
                    state = 1
        # if top is already on
        elif state_last == 2:
            if T0_h_k_thermostat > x_ref_high:
                state = 1
            else:
                state = 2

        if state == 0:
            Q_l_array[i] = 0
            Q_h_array[i] = 0
        elif state == 1:
            Q_l_array[i] = u_max_l
            Q_h_array[i] = 0
        elif state == 2:
            Q_l_array[i] = 0
            Q_h_array[i] = u_max_h

        #If mixing valve is mixing
        if temp_sim_k[i, M - 1]>=mv_set:
            flowrate_metric_h=q_array_target[i]/(rho*cp*(temp_sim_k[i, M - 1]-T_in))
            q_array_actual_array.append(q_array_target[i])
            flowrate_metric_m=flowrate_metric_h*((temp_sim_k[i, M - 1]-T_in)/(mv_set-T_in))
        #If outlet temperature is below mixing valve setpoint
        else:
            flowrate_metric_h=q_array_target[i]/(rho*cp*(mv_set-T_in))
            q_array_actual_array.append(flowrate_metric_h*rho*cp*(temp_sim_k[i, M - 1]-T_in))
            flowrate_metric_m=flowrate_metric_h

        flowrate_metric_h_array.append(flowrate_metric_h)
        flowrate_metric_m_array.append(flowrate_metric_m)

        Q_vec = np.zeros((M,))

        Q_vec[element_bot_ind] = Q_l_array[i]
        Q_vec[element_top_ind] = Q_h_array[i]
        temp_dummy=np.zeros((M,))

        #Simulate dynamics
        for j in range(M):
            diff_mod = diff * np.ones((M - 1,))

            # bottom node
            if j == 0:
                dT = ((-(A_surf + slug_cross_area * 1.2) / (cap * R)) * (temp_sim_k[i, j] - T_amb) +
                      (diff_mod[j]) * ((temp_sim_k[i, j + 1] - temp_sim_k[i, j]) / (dx ** 2)) -
                      (flowrate_metric_h / slug_cross_area) * ((temp_sim_k[i, j] - T_in) / dx) +
                      Q_vec[j] / cap )
            # top node
            elif j == M - 1:
                dT = ((-(A_surf + slug_cross_area * 1.7) / (cap * R)) * (temp_sim_k[i, j] - T_amb) +
                      (diff_mod[j - 1]) * ((temp_sim_k[i, j - 1] - temp_sim_k[i, j]) / (dx ** 2)) -
                      (flowrate_metric_h / slug_cross_area) * ((temp_sim_k[i, j] - temp_sim_k[i, j - 1]) / dx) +
                      Q_vec[j] / cap )
            #middle nodes
            else:
                dT = ((-A_surf / (cap * R)) * (temp_sim_k[i, j] - T_amb) +
                      (diff_mod[j - 1]) * ((temp_sim_k[i, j - 1] - temp_sim_k[i, j]) / (dx ** 2)) +
                      (diff_mod[j]) * ((temp_sim_k[i, j + 1] - temp_sim_k[i, j]) / (dx ** 2)) -
                      (flowrate_metric_h / slug_cross_area) * ((temp_sim_k[i, j] - temp_sim_k[i, j - 1]) / dx) +
                      Q_vec[j] / cap)

            temp_dummy[j]=temp_sim_k[i, j] + sim_dt * dT

        #Heuristic for dealing with buoyancy dynamics. Mix nodes where there are temperature inversions (up to a certain tolerance)
        bool_check = True
        while (bool_check == True):
            diff_array = np.diff(temp_dummy)

            if np.min(diff_array) < -0.01:
                zero_index = np.argmax(diff_array < -0.01)
                temp_dummy[zero_index:zero_index + 2] = (temp_dummy[zero_index] + temp_dummy[zero_index + 1]) / 2.0
                bool_check = True
            else:
                bool_check = False
        temp_sim_k[i + 1, :] = temp_dummy

        amb_loss=np.dot(loss_coef,np.ndarray.flatten(temp_sim_k[i,:])-T_amb)*sim_dt #J
        amb_loss_array.append(amb_loss)

    #Function outputs
    amb_loss_array=np.array(amb_loss_array) #Ambient losses
    flowrate_metric_h_array=np.array(flowrate_metric_h_array) #outlet flow rate
    q_array_actual_array=np.array(q_array_actual_array) #draw power
    flowrate_metric_m_array=np.array(flowrate_metric_m_array) #mixed flow rate

    return temp_sim_k,amb_loss_array,flowrate_metric_h_array,Q_l_array,Q_h_array,q_array_actual_array,state,flowrate_metric_m_array


def sim_multinode_thermostat(wh_params, sim_dt, mpc_dt, Q_l, Q_h, q_array_target, T0):

    M = wh_params['M'] #number of nodes
    element_bot_ind = wh_params['element_bot_ind'] #node index of bottom element
    element_top_ind = wh_params['element_top_ind'] #node index of top element
    diff = wh_params['diff'] #diffusion coefficient
    A_surf = wh_params['A_surf'] #surface area
    slug_cross_area = wh_params['slug_cross_area'] #cross sectional area
    cap = wh_params['cap'] #thermal capacitance
    R = wh_params['R'] #thermal resistance of insulation
    T_amb = wh_params['T_amb'] #ambient temperature
    dx = wh_params['dx'] #node height
    T_in = wh_params['T_in'] #inlet temperature [K]
    mv_set_f = wh_params['mv_set_f'] #mixing valve setpoint
    mv_set = (mv_set_f - 32) * (5.0 / 9.0) + 273.15
    rho=wh_params['rho']
    cp=wh_params['cp']

    N_sim = int(mpc_dt / sim_dt)
    temp_sim_k = np.zeros((N_sim + 1, M))
    temp_sim_k[0, :] = T0

    loss_coef=np.ones((M,))*(A_surf/R)
    loss_coef[0]=(A_surf+slug_cross_area*1.2)/R
    loss_coef[M-1]=(A_surf+slug_cross_area*1.7)/R

    amb_loss_array=[]
    flowrate_metric_h_array=[]
    q_array_actual_array=[]
    flowrate_metric_m_array=[]

    #loop through timesteps
    for i in range(N_sim):

        #if outlet temperature is about mixing valve setpoint, the valve mixes hot and cold water
        if temp_sim_k[i, M - 1]>=mv_set:
            flowrate_metric_h=q_array_target[i]/(rho*cp*(temp_sim_k[i, M - 1]-T_in))
            q_array_actual_array.append(q_array_target[i])
            flowrate_metric_m=flowrate_metric_h*((temp_sim_k[i, M - 1]-T_in)/(mv_set-T_in))
        #if the outlet temperature is below the mixing valve setpoint
        else:
            flowrate_metric_h=q_array_target[i]/(rho*cp*(mv_set-T_in))
            q_array_actual_array.append(flowrate_metric_h*rho*cp*(temp_sim_k[i, M - 1]-T_in))
            flowrate_metric_m=flowrate_metric_h

        flowrate_metric_h_array.append(flowrate_metric_h)
        flowrate_metric_m_array.append(flowrate_metric_m)


        Q_vec = np.zeros((M,))

        Q_vec[element_bot_ind] = Q_l
        Q_vec[element_top_ind] = Q_h
        temp_dummy=np.zeros((M,))

        #simulate thermal dynamics
        for j in range(M):
            diff_mod = diff * np.ones((M - 1,))

            # bottom node
            if j == 0:
                dT = ((-(A_surf + slug_cross_area * 1.2) / (cap * R)) * (temp_sim_k[i, j] - T_amb) +
                      (diff_mod[j]) * ((temp_sim_k[i, j + 1] - temp_sim_k[i, j]) / (dx ** 2)) -
                      (flowrate_metric_h / slug_cross_area) * ((temp_sim_k[i, j] - T_in) / dx) +
                      Q_vec[j] / cap )
            # top node
            elif j == M - 1:
                dT = ((-(A_surf + slug_cross_area * 1.7) / (cap * R)) * (temp_sim_k[i, j] - T_amb) +
                      (diff_mod[j - 1]) * ((temp_sim_k[i, j - 1] - temp_sim_k[i, j]) / (dx ** 2)) -
                      (flowrate_metric_h / slug_cross_area) * ((temp_sim_k[i, j] - temp_sim_k[i, j - 1]) / dx) +
                      Q_vec[j] / cap )
            #middle nodes
            else:
                dT = ((-A_surf / (cap * R)) * (temp_sim_k[i, j] - T_amb) +
                      (diff_mod[j - 1]) * ((temp_sim_k[i, j - 1] - temp_sim_k[i, j]) / (dx ** 2)) +
                      (diff_mod[j]) * ((temp_sim_k[i, j + 1] - temp_sim_k[i, j]) / (dx ** 2)) -
                      (flowrate_metric_h / slug_cross_area) * ((temp_sim_k[i, j] - temp_sim_k[i, j - 1]) / dx) +
                      Q_vec[j] / cap)

            temp_dummy[j]=temp_sim_k[i, j] + sim_dt * dT

        #Heuristic for buoyancy dynamics: mix nodes that have temperature inversions
        bool_check = True
        while (bool_check == True):
            diff_array = np.diff(temp_dummy)

            if np.min(diff_array) < -0.01:
                zero_index = np.argmax(diff_array < -0.01)
                temp_dummy[zero_index:zero_index + 2] = (temp_dummy[zero_index] + temp_dummy[zero_index + 1]) / 2.0
                bool_check = True
            else:
                bool_check = False

        temp_sim_k[i + 1, :] = temp_dummy

        amb_loss=np.dot(loss_coef,np.ndarray.flatten(temp_sim_k[i,:])-T_amb)*sim_dt #J
        amb_loss_array.append(amb_loss)

    amb_loss_array=np.array(amb_loss_array) #ambient losses
    flowrate_metric_h_array=np.array(flowrate_metric_h_array) #flow rate of outlet
    q_array_actual_array=np.array(q_array_actual_array) #draw power
    flowrate_metric_m_array=np.array(flowrate_metric_m_array) #flow rate of mixed water

    return temp_sim_k,amb_loss_array,flowrate_metric_h_array,q_array_actual_array,flowrate_metric_m_array


def sim_multinode_mpc(wh_params, sim_dt, mpc_dt, Q_l, Q_h, q_array_target, T0,Q_l_nom,Q_h_nom,f_model_params,model_dir):

    #controller parameters
    os.chdir(model_dir)
    f=open(f_model_params,'r')
    model_param_dict=f.read()
    f.close()
    model_param_dict=ast.literal_eval(model_param_dict)

    M = wh_params['M'] #number of nodes
    element_bot_ind = wh_params['element_bot_ind'] #node index of bottom element
    element_top_ind = wh_params['element_top_ind'] #node index of top element
    diff = wh_params['diff'] #diffusion coefficient
    A_surf = wh_params['A_surf'] #surface area
    slug_cross_area = wh_params['slug_cross_area'] #cross sectional area
    cap = wh_params['cap'] #thermal capacitance
    R = wh_params['R'] #thermal resistance of insulation
    T_amb = wh_params['T_amb'] # ambient temperature
    dx = wh_params['dx'] #node height
    T_in = wh_params['T_in'] #inlet temperature
    mv_set_f = wh_params['mv_set_f'] #mixing valve setpoint
    rho=wh_params['rho'] #water density
    cp=wh_params['cp'] #thermal heat capacity of water
    mv_set = (mv_set_f - 32) * (5.0 / 9.0) + 273.15
    x_ref_high=(model_param_dict['x_ref_high_f'] - 32) * (5.0 / 9.0) + 273.15 #max temperature

    N_sim = int(mpc_dt / sim_dt)
    temp_sim_k = np.zeros((N_sim + 1, M))
    temp_sim_k[0, :] = T0

    loss_coef=np.ones((M,))*(A_surf/R)
    loss_coef[0]=(A_surf+slug_cross_area*1.2)/R
    loss_coef[M-1]=(A_surf+slug_cross_area*1.7)/R

    amb_loss_array=[]
    flowrate_metric_h_array=[]
    q_array_actual_array=[]
    flowrate_metric_m_array=[]

    #convert continuous power value to on/off signal
    Q_l_array=np.zeros((N_sim,))
    Q_h_array=np.zeros((N_sim,))
    Q_l_start=0
    Q_l_duty=Q_l/Q_l_nom
    Q_h_duty=Q_h/Q_h_nom
    if Q_l_duty<0.05:
        Q_l_duty=0.0
    if Q_h_duty<0.05:
        Q_h_duty=0.0
    Q_l_end=int(Q_l_duty*N_sim)
    Q_h_start=Q_l_end
    Q_h_end=int(Q_h_start+Q_h_duty*N_sim)
    Q_l_array[Q_l_start:Q_l_end]=Q_l_nom
    Q_h_array[Q_h_start:Q_h_end]=Q_h_nom

    #loop through timesteps
    for i in range(N_sim):

        #if outlet temperature is above mixing valve setpoint, the valve mixes hot and cold water
        if temp_sim_k[i, M - 1]>=mv_set:
            flowrate_metric_h=q_array_target[i]/(rho*cp*(temp_sim_k[i, M - 1]-T_in))
            q_array_actual_array.append(q_array_target[i])
            flowrate_metric_m=flowrate_metric_h*((temp_sim_k[i, M - 1]-T_in)/(mv_set-T_in))
        #if outlet temperature is blow mixing valve setpoint
        else:
            flowrate_metric_h=q_array_target[i]/(rho*cp*(mv_set-T_in))
            q_array_actual_array.append(flowrate_metric_h*rho*cp*(temp_sim_k[i, M - 1]-T_in))
            flowrate_metric_m=flowrate_metric_h

        flowrate_metric_h_array.append(flowrate_metric_h)
        flowrate_metric_m_array.append(flowrate_metric_m)

        Q_vec = np.zeros((M,))

        if x_ref_high<temp_sim_k[i, M - 1]:
            Q_l_array[i]=0
            Q_h_array[i]=0

        Q_vec[element_bot_ind] = Q_l_array[i]
        Q_vec[element_top_ind] = Q_h_array[i]
        temp_dummy=np.zeros((M,))

        #simulate thermal dynamics
        for j in range(M):
            diff_mod = diff * np.ones((M - 1,))

            # bottom node
            if j == 0:
                dT = ((-(A_surf + slug_cross_area * 1.2) / (cap * R)) * (temp_sim_k[i, j] - T_amb) +
                      (diff_mod[j]) * ((temp_sim_k[i, j + 1] - temp_sim_k[i, j]) / (dx ** 2)) -
                      (flowrate_metric_h / slug_cross_area) * ((temp_sim_k[i, j] - T_in) / dx) +
                      Q_vec[j] / cap )
            # top node
            elif j == M - 1:
                dT = ((-(A_surf + slug_cross_area * 1.7) / (cap * R)) * (temp_sim_k[i, j] - T_amb) +
                      (diff_mod[j - 1]) * ((temp_sim_k[i, j - 1] - temp_sim_k[i, j]) / (dx ** 2)) -
                      (flowrate_metric_h / slug_cross_area) * ((temp_sim_k[i, j] - temp_sim_k[i, j - 1]) / dx) +
                      Q_vec[j] / cap )
            #middle node
            else:
                dT = ((-A_surf / (cap * R)) * (temp_sim_k[i, j] - T_amb) +
                      (diff_mod[j - 1]) * ((temp_sim_k[i, j - 1] - temp_sim_k[i, j]) / (dx ** 2)) +
                      (diff_mod[j]) * ((temp_sim_k[i, j + 1] - temp_sim_k[i, j]) / (dx ** 2)) -
                      (flowrate_metric_h / slug_cross_area) * ((temp_sim_k[i, j] - temp_sim_k[i, j - 1]) / dx) +
                      Q_vec[j] / cap)

            temp_dummy[j]=temp_sim_k[i, j] + sim_dt * dT

        #heuristic for modeling buoyancy dynamics: mix nodes with temperature inversions
        bool_check = True
        while (bool_check == True):
            diff_array = np.diff(temp_dummy)

            if np.min(diff_array) < -0.01:
                zero_index = np.argmax(diff_array < -0.01)
                temp_dummy[zero_index:zero_index + 2] = (temp_dummy[zero_index] + temp_dummy[zero_index + 1]) / 2.0
                bool_check = True
            else:
                bool_check = False

        temp_sim_k[i + 1, :] = temp_dummy

        amb_loss=np.dot(loss_coef,np.ndarray.flatten(temp_sim_k[i,:])-T_amb)*sim_dt #J
        amb_loss_array.append(amb_loss)

    amb_loss_array=np.array(amb_loss_array) #ambient losses
    flowrate_metric_h_array=np.array(flowrate_metric_h_array) #outlet water flow rate
    flowrate_metric_m_array=np.array(flowrate_metric_m_array) # mixed water flow rate
    q_array_actual_array=np.array(q_array_actual_array) #draw power

    return temp_sim_k,amb_loss_array,flowrate_metric_h_array,Q_l_array,Q_h_array,q_array_actual_array,flowrate_metric_m_array


