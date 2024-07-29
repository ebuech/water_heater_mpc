import numpy as np
import time
import os
import datetime
import calendar
import pandas
import ast


def thermostat(T0_m_f,T0_h_f,state_last,f_model_params,model_dir):

    #controller parameters
    os.chdir(model_dir)
    f=open(f_model_params,'r')
    model_param_dict=f.read()
    f.close()
    model_param_dict=ast.literal_eval(model_param_dict)

    x_ref_low_f=model_param_dict['x_ref_low_f'] #lower bound on deadband
    x_ref_high_f=model_param_dict['x_ref_high_f'] #upper bound on deadband
    u_max_l=model_param_dict['u_max_l'] #max power lower element
    u_max_h=model_param_dict['u_max_h'] #max power upper element

    T0_m=(T0_m_f-32.0)*(5.0/9.0)+273.15 #initial temperature
    T0_h=(T0_h_f-32.0)*(5.0/9.0)+273.15

    x_ref_low=(x_ref_low_f-32.0)*(5.0/9.0)+273.15
    x_ref_high=(x_ref_high_f-32.0)*(5.0/9.0)+273.15

    #thermostatic control logic
    # if top isnt already on
    if (state_last==1) or (state_last==0):
        #turn on if need be
        if T0_h<=x_ref_low:
            state=2
        # if bottom is off
        elif state_last==0:
            #see if it needs to be turned on
            if T0_m<=x_ref_low:
                state=1
            else:
                state=0
        # if bottom is on
        elif state_last==1:
            if T0_m>x_ref_high:
                state=0
            else:
                state=1
    #if top is already on
    elif state_last==2:
        if T0_h>x_ref_high:
            state=1
        else:
            state=2

    if state==0:
        ul_opt=0
        uh_opt=0
    elif state==1:
        ul_opt=u_max_l
        uh_opt=0
    elif state==2:
        ul_opt=0
        uh_opt=u_max_h

    return ul_opt,uh_opt,state

