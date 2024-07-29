
import os
cwd=os.getcwd()
import sys
sys.path.append(os.path.join(cwd, 'functions'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import matplotlib
import ast
from casadi import *
#sys.path.append('/Users/lilybuechler/Documents/stanford/research/water_heater_controls/sdf_control/util')
import multinode_sim
import mpc
#import thermostat
import copy

######################################################################
##############        Define scenario parameters        ##############
######################################################################

T=3600*24*56 # Simulation horizon [seconds]
days_of_simulation=T/(3600*24)
mpc_dt=600 #MPC timestep [seconds]
sim_dt=10 #multinode simulation timestep [seconds]
horizon_hr=24 #MPC optimization horizon [hrs]
mpc_T=horizon_hr*3600 #MPC optimization horizon [seconds]
mpc_M=2 #[integraton steps per control step]

model_type='1nodemv' #MPC formulation [1nodemv or 3nodemv]
ritchie_num=42 #Home number from ritchie draw data
flow_type='ritchie_'+str(int(ritchie_num))
sensor_config='5avg' #Sensor configuration [[1low, 2avg, 5avg] for 1nodemv [3act, 6avg] for 3nodemv]
forecast_type='hist_avg_fest' #Draw forecasting appraoch ['perfect','hist_avg','hist_quantile','hist_avg_fest','hist_quantile_fest']
save_fig=True #Turn figure saving on/off
n_samp=28 #Length of lagging dataset for generating draw forecasts
quant=0.9 #quantile for quantile-based draw forecasts
cost_name='TPDP' #Electricity cost profile ['elec' for PG&E TOU, 'TPDP' for CalFlexHub dynamic rate]
f_multinode_params='multinode.txt'

model_dir=os.path.join(cwd, 'data/model_parameters')
out_dir=os.path.join(cwd,'results')

N=int(mpc_T/mpc_dt)
N_total=int(T/mpc_dt)


######################################################################
###########        load electricity cost profile        ##############
######################################################################

if cost_name=='elec':
    cost_filename='pge_'+cost_name+'_60day.csv'
if cost_name=='TPDP':
    cost_filename='MIDAS_rate_60day_USCA-CFCF-'+cost_name+'-0000.csv'

os.chdir(os.path.join(cwd, 'data/electricity_cost'))
cost_sch=pd.read_csv(cost_filename)
c_array_total=np.zeros((N_total+N),)
for i in range(cost_sch.shape[0]):
    c_array_total[int(cost_sch.time[i]/mpc_dt):int((cost_sch.time[i]/mpc_dt)+(cost_sch.duration[i]/mpc_dt))]=float(cost_sch.cost[i])
c_array_full=np.zeros(int(T/sim_dt),)
for i in range(cost_sch.shape[0]):
    c_array_full[int(cost_sch.time[i]/sim_dt):int((cost_sch.time[i]/sim_dt)+(cost_sch.duration[i]/sim_dt))]=float(cost_sch.cost[i])


######################################################################
#################        Load flow rate data        ##################
######################################################################

os.chdir(os.path.join(cwd, 'data/Raw Water Profiles'))
data=pd.read_csv('ewh_profile['+str(int(ritchie_num))+'].csv')

#Load the four one-month chunks of data and concatenate, as if they were consecutive
data_summer=data[['Summer_Timestamps','Summer_Water_Consumption']] #2/1/18-2/28/18
data_winter=data[['Winter_Timestamps','Winter_Water_Consumption']] #7/25/17-8/21/17
data_spring=data[['Spring_Timestamps','Spring_Water_Consumption']] #9/5/17 - 10/2/17
data_autumn=data[['Autumn_Timestamps','Autumn_Water_Consumption']] #3/13/18 - 4/9/18

data_winter['timestamp']=pd.DatetimeIndex(data_winter['Winter_Timestamps'])
data_winter=data_winter.set_index(data_winter['timestamp'])
data_winter2=data_winter.sort_index()

t0=data_winter2.index[-1]+datetime.timedelta(minutes=1)
t_list=[]
for i in range(60*24*28):
    t_list.append(t0)
    t0=t0+datetime.timedelta(minutes=1)

data_spring['timestamp']=pd.DatetimeIndex(data_spring['Spring_Timestamps'])
data_spring=data_spring.set_index(data_spring['timestamp'])
data_spring2=data_spring.sort_index()
data_spring3=pd.DataFrame({'timestamp':t_list,'Winter_Water_Consumption':np.array(data_spring2.Spring_Water_Consumption)})
data_spring3=data_spring3.set_index(data_spring3['timestamp'])

t_list=[]
for i in range(60*24*28):
    t_list.append(t0)
    t0=t0+datetime.timedelta(minutes=1)

data_summer['timestamp']=pd.DatetimeIndex(data_summer['Summer_Timestamps'])
data_summer=data_summer.set_index(data_summer['timestamp'])
data_summer2=data_summer.sort_index()
data_summer3=pd.DataFrame({'timestamp':t_list,'Winter_Water_Consumption':np.array(data_summer2.Summer_Water_Consumption)})
data_summer3=data_summer3.set_index(data_summer3['timestamp'])

t_list=[]
for i in range(60*24*28):
    t_list.append(t0)
    t0=t0+datetime.timedelta(minutes=1)

data_autumn['timestamp']=pd.DatetimeIndex(data_autumn['Autumn_Timestamps'])
data_autumn=data_autumn.set_index(data_autumn['timestamp'])
data_autumn2=data_autumn.sort_index()
data_autumn3=pd.DataFrame({'timestamp':t_list,'Winter_Water_Consumption':np.array(data_autumn2.Autumn_Water_Consumption)})
data_autumn3=data_autumn3.set_index(data_autumn3['timestamp'])

data_full=pd.concat((data_winter2,data_spring3,data_summer3,data_autumn3),axis=0)
data_full=data_full[['Winter_Water_Consumption']]
data_full_raw_10min=data_full.resample('10min').sum()
data_full_raw_10min['m3s']=data_full_raw_10min.Winter_Water_Consumption*0.00378541/3.78541/mpc_dt

data_full_hour=data_full.resample('H').sum()
data_full_hour['hour']=data_full_hour.index.hour
data_full_10min=data_full_hour.resample('10min').ffill()
data_full_10min['m3s']=data_full_10min.Winter_Water_Consumption*0.00378541/3.78541/3600
data_full_10min['control_step']=data_full_10min.index.hour*(int(3600/mpc_dt))+(data_full_10min.index.minute*60/mpc_dt)


data_full_test=data_full[(data_full.index>=datetime.datetime(2017,7,25))&(data_full.index<datetime.datetime(2017,9,20))]
f_array_test_min=np.array(data_full_test.Winter_Water_Consumption)*(0.00378541/3.78541)/60
f_array_total=np.ndarray.flatten(np.tile(f_array_test_min.reshape(-1,1),(1,int(60/sim_dt))).reshape(-1,1))[0:int(T/sim_dt)+int(mpc_T/sim_dt)]

f_array_reshape=f_array_total.reshape(-1,int(mpc_dt/sim_dt))
f_array_mean=np.mean(f_array_reshape,axis=1)
f_array_mean=f_array_mean.reshape(-1,1)


######################################################################
############    Define MPC control model parameters     ##############
######################################################################

#control model parameters
if '1nodemv'==model_type:
    f_model_params='model_1node_mv_240V.txt'
elif ('3nodemv'== model_type):
    f_model_params='model_3node_mv_240V.txt'
#thermostatic controller parameters
f_model_params_thermostat='model_thermostat_mv_240V.txt'

#Load file and parse as dictionary
os.chdir(model_dir)
f=open(f_model_params,'r')
model_param_dict=f.read()
f.close()
model_param_dict=ast.literal_eval(model_param_dict)


######################################################################
###########    Define multi-node model parameters     ################
######################################################################

os.chdir(model_dir)
f=open(f_multinode_params,'r')
wh_params=f.read()
f.close()
wh_params=ast.literal_eval(wh_params)

sensor_heights=np.array(wh_params['sensor_heights_in'])/39.37 #[m]
h=wh_params['h_in']/39.37 #[m]
sensor_slug_height=np.array([(sensor_heights[0]+sensor_heights[1])/2,
                             ((sensor_heights[1]+sensor_heights[2])/2)-((sensor_heights[0]+sensor_heights[1])/2),
                             ((sensor_heights[2]+sensor_heights[3])/2)-((sensor_heights[1]+sensor_heights[2])/2),
                             ((sensor_heights[3]+sensor_heights[4])/2)-((sensor_heights[2]+sensor_heights[3])/2),
                             ((sensor_heights[4]+sensor_heights[5])/2)-((sensor_heights[3]+sensor_heights[4])/2),
                             h-((sensor_heights[4]+sensor_heights[5])/2)])
sensor_bins=np.linspace(0,h,wh_params['M']+1)
sensor_ind=np.digitize(sensor_heights,sensor_bins)-1 #nodes that actual sensor locations correspond to

element_height_top=wh_params['element_height_top_in']/39.37 #Height of upper element
element_height_bot=wh_params['element_height_bot_in']/39.37 #Height of lower element

r=wh_params['r_in']/39.37 #[m]
dx=h/wh_params['M'] #[m]
A_surf=dx*np.pi*r*2 #[m^2] surface area
slug_cross_area=np.pi*(r**2)
slug_vol=dx*slug_cross_area #[m^3] node volume
cap=slug_vol*wh_params['cp']*wh_params['rho']

#temperature array for simulation
temp_sim_k = np.zeros((int(T/sim_dt),wh_params['M']))
#Initial tank profile [for 20 node model]. Initializes tank around 120 deg F
temp_sim_k[0,:]=np.array(wh_params['T_init_prof'])

#collect multi-node model parameters
wh_params['element_bot_ind'] = np.digitize(element_height_bot,sensor_bins)-1
wh_params['element_top_ind'] = np.digitize(element_height_top,sensor_bins)-1
wh_params['diff'] = wh_params['k']/(wh_params['cp']*wh_params['rho'])
wh_params['A_surf'] = A_surf
wh_params['slug_cross_area'] = slug_cross_area
wh_params['cap'] = cap
wh_params['T_amb'] = (wh_params['T_amb_f']-32.0)*(5.0/9.0)+273.15
wh_params['dx'] = dx
wh_params['T_in'] = (wh_params['T_in_f']-32.0)*(5.0/9.0)+273.15
T_in=wh_params['T_in']
mv_set_k=(wh_params['mv_set_f']-32.0)*(5.0/9.0)+273.15 #Mixing valve setpoint [K]
T_amb=(wh_params['T_amb_f']-32.0)*(5.0/9.0)+273.15 #Ambient temperature [K]
M=wh_params['M']


######################################################################
#################    Target draw power profile     ##################
######################################################################

#Target Q^{(d)} values
q_array_total=f_array_total*model_param_dict['rho']*model_param_dict['cp']*(mv_set_k-T_in)
q_array_mean=f_array_mean*model_param_dict['rho']*model_param_dict['cp']*(mv_set_k-T_in)
data_full_10min['q_draw']=data_full_10min.m3s*model_param_dict['rho']*model_param_dict['cp']*(mv_set_k-T_in)

######################################################################
###################      Control simulation       ####################
######################################################################

state = 0 #thermostatic controller state
t_now = 0
init_bool = False
prev_sol = {}
iter_count = [] #track MPC solver iterations
time_array = []
t_datetime=datetime.datetime(2017,7,25) #Date of start of the simulation
t_datetime_mpc_start=t_datetime+datetime.timedelta(days=28) #date when MPC controller turns on

Q_array_cl = np.zeros((int(T/sim_dt), 2)) #array for logging power consumption
flowrate_mod=np.zeros((int(T/sim_dt),)) #Array for logging outlet flow rate
flowrate_mixed=np.zeros((int(T/sim_dt),)) #Array for logging mixed flow rate
q_array_total_actual=np.zeros((int(T/sim_dt),)) #Array for logging actual draw power values

#setup MPC formulation
if '1nodemv'==model_type:
    solver,solver_def=mpc.setup_1node_mv(mpc_M,N,mpc_T,f_model_params,model_dir)
    init_bool=False
    prev_sol={}
elif ('3nodemv'==model_type) or ('3nodemvlf'==model_type):
    if model_type=='3nodemv':
        solver,solver_def=mpc.setup_3node_mv(mpc_M,N,mpc_T,f_model_params,model_dir)
    init_bool=False
    prev_sol={}

#Array for recording computation time
comp_time=np.zeros((int(T/mpc_dt),))
#Array for recording estimated draws
est_flow_power=np.zeros((int(T/mpc_dt),))

#step through control timesteps
for i in range(int(T / mpc_dt)):
    print(str(i)+' out of '+str(int(T/mpc_dt)))
    t1 = time.time()
    #Current temperature
    temp_now = np.array(temp_sim_k[int(t_now/sim_dt), :])
    temp_now_samp = temp_now[sensor_ind] #temperature at sensor locations
    temp_now_samp_F = (temp_now_samp - 273.15) * (9.0 / 5.0) + 32.0

    t1=time.time()

    if model_type=='1nodemv':

        c_array=c_array_total[i+1:i+1+N] #cost profile over MPC horizon

        #measure temperature
        if sensor_config=='1low':
            T0_f=temp_now_samp_F[6]
            T0_f_safety=temp_now_samp_F[6]
        if sensor_config=='2avg':
            T0_f=(temp_now_samp_F[6]+temp_now_samp_F[7])*0.5
            T0_f_safety=temp_now_samp_F[7]
        if sensor_config=='5avg':
            T0_f=(temp_now_samp_F[1]+temp_now_samp_F[2]+temp_now_samp_F[3]+temp_now_samp_F[4]+temp_now_samp_F[5])/5.0
            T0_f_safety=temp_now_samp_F[5]
        T0_m_f_thermostat = temp_now_samp_F[6]
        T0_h_f_thermostat = temp_now_samp_F[7]

        #estimate last water draw
        if i!=0:
            T0_k = (T0_f - 32.0) * (5.0 / 9.0) + 273.15
            t_datetime_last=t_datetime - datetime.timedelta(seconds=mpc_dt)
            #Water draw estimation
            est_flow_power[i-1]=((model_param_dict['U_1node']*(T_amb-T0_k_last))-(model_param_dict['vol_1node']*wh_params['cp']*wh_params['rho']*(T0_k-T0_k_last)/mpc_dt)+Q_l_last)
            #Add to data frame
            if i==1:
                est_flow_power_df=pd.DataFrame({'hour':t_datetime_last.hour,'q_draw':est_flow_power[i-1],'control_step':t_datetime_last.hour*(int(3600/mpc_dt))+(t_datetime_last.minute*60/mpc_dt)},index=[pd.Timestamp(t_datetime-datetime.timedelta(seconds=mpc_dt))])
            else:
                est_flow_power_df=pd.concat((est_flow_power_df,pd.DataFrame({'hour':t_datetime_last.hour,'q_draw':est_flow_power[i-1],'control_step':t_datetime_last.hour*(int(3600/mpc_dt))+(t_datetime_last.minute*60/mpc_dt)},index=[pd.Timestamp(t_datetime-datetime.timedelta(seconds=mpc_dt))])))

        if t_datetime>=t_datetime_mpc_start: #run mpc after thermostatic control is done running

            #forecast flow across horizon
            if (forecast_type=='hist_avg') or (forecast_type=='hist_quantile'): #forecasting based on actual draws
                actual_flow_power_df_10min_avg = copy.copy(actual_flow_power_df).resample('10min').mean()
                actual_flow_power_df_10min_avg['control_step'] = actual_flow_power_df_10min_avg.index.hour * (int(3600 / mpc_dt)) + (actual_flow_power_df_10min_avg.index.minute * 60 / mpc_dt)
                actual_flow_power_df_10min_avg=actual_flow_power_df_10min_avg[(actual_flow_power_df_10min_avg.index<t_datetime)&(actual_flow_power_df_10min_avg.index>=t_datetime-datetime.timedelta(days=n_samp))]
                if (forecast_type == 'hist_avg'):
                    actual_flow_power_df_10min_avg2 = actual_flow_power_df_10min_avg.groupby('control_step',sort=False).mean()
                elif (forecast_type=='hist_quantile'):
                    actual_flow_power_df_10min_avg2 = actual_flow_power_df_10min_avg.groupby('control_step',sort=False).quantile(quant)
                q_array = np.maximum(np.array(actual_flow_power_df_10min_avg2.q_draw)[0:N],0)

            elif forecast_type=='perfect': #perfect foresight
                q_array = q_array_mean[i + 1:i + 1 + N]

            elif (forecast_type=='hist_avg_fest') or (forecast_type=='hist_quantile_fest'): #forecasting based on estimated draws
                est_flow_power_df_10min_avg = copy.copy(est_flow_power_df).resample('10min').mean()
                est_flow_power_df_10min_avg['control_step'] = est_flow_power_df_10min_avg.index.hour * (int(3600 / mpc_dt)) + (est_flow_power_df_10min_avg.index.minute * 60 / mpc_dt)
                est_flow_power_df_10min_avg=est_flow_power_df_10min_avg[(est_flow_power_df_10min_avg.index<t_datetime)&(est_flow_power_df_10min_avg.index>=t_datetime-datetime.timedelta(days=n_samp))]
                if (forecast_type == 'hist_avg_fest'):
                    est_flow_power_df_10min_avg2 = est_flow_power_df_10min_avg.groupby('control_step',sort=False).mean()
                elif (forecast_type=='hist_quantile_fest'):
                    est_flow_power_df_10min_avg2 = est_flow_power_df_10min_avg.groupby('control_step',sort=False).quantile(quant)
                q_array = np.maximum(np.array(est_flow_power_df_10min_avg2.q_draw)[0:N],0)

            # solve MPC if tank isn't too hot
            if T0_f_safety<model_param_dict['x_ref_high_f']:
                u_opt,x_opt_f,prev_sol,it_count=mpc.solve_1node_mv(solver,solver_def,T0_f,q_array,c_array,init_bool,prev_sol)

                init_bool=True
                Q_l = np.round(float(u_opt[0]), 2)
                Q_h=0.0
            else: #otherwise turn off the elements
                Q_l=0.0
                Q_h=0.0
            thermostat_bool=False
        else: #run thermostat
            thermostat_bool=True

    ########## 3 node MPC ############
    elif (model_type=='3nodemv'):

        c_array=c_array_total[i+1:i+1+N] #Cost profile over optimization horizon

        #measure temperature
        if sensor_config=='3act':
            T0_l_f=temp_now_samp_F[0]
            T0_m_f=temp_now_samp_F[6]
            T0_h_f=temp_now_samp_F[7]
        if sensor_config=='6avg':
            T0_l_f=temp_now_samp_F[0]
            T0_m_f=(temp_now_samp_F[1]+temp_now_samp_F[2]+temp_now_samp_F[3])/3.0
            T0_h_f=(temp_now_samp_F[4]+temp_now_samp_F[5])/2.0
        T0_m_f_thermostat = temp_now_samp_F[6]
        T0_h_f_thermostat = temp_now_samp_F[7]

        #estimate last water draw
        if i != 0:
            T0_l_k = (T0_l_f - 32.0) * (5.0 / 9.0) + 273.15
            T0_m_k = (T0_m_f - 32.0) * (5.0 / 9.0) + 273.15
            T0_h_k = (T0_h_f - 32.0) * (5.0 / 9.0) + 273.15
            t_datetime_last=t_datetime - datetime.timedelta(seconds=mpc_dt)

            est_flow_power[i - 1] = ((model_param_dict['U_l_3node'] * (T_amb - T0_l_k_last))+(model_param_dict['U_m_3node'] * (T_amb - T0_m_k_last))+(model_param_dict['U_h_3node'] * (T_amb - T0_h_k_last))
                                      - (model_param_dict['vol_l_3node'] * wh_params['cp'] * wh_params['rho'] * (T0_l_k - T0_l_k_last) / float(mpc_dt))-(model_param_dict['vol_m_3node'] * wh_params['cp'] * wh_params['rho'] * (T0_m_k - T0_m_k_last) / float(mpc_dt))-(model_param_dict['vol_h_3node'] * wh_params['cp'] * wh_params['rho'] * (T0_h_k - T0_h_k_last) / float(mpc_dt))
                                      + Q_l_last+Q_h_last)
            if i==1:
                est_flow_power_df=pd.DataFrame({'hour':t_datetime_last.hour,'q_draw':est_flow_power[i-1],'control_step':t_datetime_last.hour*(int(3600/mpc_dt))+(t_datetime_last.minute*60/mpc_dt)},index=[pd.Timestamp(t_datetime-datetime.timedelta(seconds=mpc_dt))])
            else:
                est_flow_power_df=pd.concat((est_flow_power_df,pd.DataFrame({'hour':t_datetime_last.hour,'q_draw':est_flow_power[i-1],'control_step':t_datetime_last.hour*(int(3600/mpc_dt))+(t_datetime_last.minute*60/mpc_dt)},index=[pd.Timestamp(t_datetime-datetime.timedelta(seconds=mpc_dt))])))


        if t_datetime>=t_datetime_mpc_start: #run mpc

            #forecast flow across horizon
            if (forecast_type=='hist_avg') or (forecast_type=='hist_quantile'):
                actual_flow_power_df_10min_avg = copy.copy(actual_flow_power_df).resample('10min').mean()
                actual_flow_power_df_10min_avg['control_step'] = actual_flow_power_df_10min_avg.index.hour * (int(3600 / mpc_dt)) + (actual_flow_power_df_10min_avg.index.minute * 60 / mpc_dt)
                actual_flow_power_df_10min_avg=actual_flow_power_df_10min_avg[(actual_flow_power_df_10min_avg.index<t_datetime)&(actual_flow_power_df_10min_avg.index>=t_datetime-datetime.timedelta(days=n_samp))]
                if (forecast_type == 'hist_avg'):
                    actual_flow_power_df_10min_avg2 = actual_flow_power_df_10min_avg.groupby('control_step',sort=False).mean()
                elif (forecast_type=='hist_quantile'):
                    actual_flow_power_df_10min_avg2 = actual_flow_power_df_10min_avg.groupby('control_step',sort=False).quantile(quant)
                q_array = np.maximum(np.array(actual_flow_power_df_10min_avg2.q_draw)[0:N],0)

            elif forecast_type=='perfect': #Perfect foresight
                q_array = q_array_mean[i + 1:i + 1 + N]

            elif (forecast_type=='hist_avg_fest') or (forecast_type=='hist_quantile_fest'):
                est_flow_power_df_10min_avg = copy.copy(est_flow_power_df).resample('10min').mean()
                est_flow_power_df_10min_avg['control_step'] = est_flow_power_df_10min_avg.index.hour * (int(3600 / mpc_dt)) + (est_flow_power_df_10min_avg.index.minute * 60 / mpc_dt)
                est_flow_power_df_10min_avg=est_flow_power_df_10min_avg[(est_flow_power_df_10min_avg.index<t_datetime)&(est_flow_power_df_10min_avg.index>=t_datetime-datetime.timedelta(days=n_samp))]
                if (forecast_type == 'hist_avg_fest'):
                    est_flow_power_df_10min_avg2 = est_flow_power_df_10min_avg.groupby('control_step',sort=False).mean()
                elif (forecast_type=='hist_quantile_fest'):
                    est_flow_power_df_10min_avg2 = est_flow_power_df_10min_avg.groupby('control_step',sort=False).quantile(quant)
                q_array = np.maximum(np.array(est_flow_power_df_10min_avg2.q_draw)[0:N],0)

            #solve MPC
            if T0_h_f<model_param_dict['x_ref_high_f']:
                ul_opt,uh_opt,xl_opt_f,xm_opt_f,xh_opt_f,prev_sol,it_count=mpc.solve_3node_mv(solver,solver_def,T0_l_f,T0_m_f,T0_h_f,q_array,c_array,init_bool,prev_sol)
                init_bool=True
                Q_l = np.round(float(ul_opt[0]), 2)
                Q_h = np.round(float(uh_opt[0]), 2)

            else:
                Q_l=0.0
                Q_h=0.0
            thermostat_bool=False
        else:
            thermostat_bool=True

    t2=time.time()
    comp_time[i]=t2-t1

    q_array_pde = (q_array_total[int(i * mpc_dt / sim_dt):int((i + 1) * mpc_dt / sim_dt)])

    if thermostat_bool==False:
        if model_type=='1nodemv':
            Q_l_nom=model_param_dict['u_max']
            Q_h_nom=model_param_dict['u_max']
        elif (model_type=='3nodemv'):
            Q_l_nom=model_param_dict['u_max_l']
            Q_h_nom=model_param_dict['u_max_h']
        temp_sim_kx,amb_loss_kx,flowrate_mod_kx,Q_l_kx,Q_h_kx,q_array_pde_act_kx,flowrate_m_kx = multinode_sim.sim_multinode_mpc(wh_params, sim_dt, mpc_dt, Q_l, Q_h, q_array_pde, temp_now,Q_l_nom,Q_h_nom,f_model_params,model_dir)
    elif thermostat_bool==True:
        temp_sim_kx,amb_loss_kx,flowrate_mod_kx,Q_l_kx,Q_h_kx,q_array_pde_act_kx,state,flowrate_m_kx = multinode_sim.sim_multinode_mpc_thermostat(wh_params, sim_dt, mpc_dt, q_array_pde, temp_now,state,f_model_params,model_dir,sensor_ind)
        Q_l=np.mean(Q_l_kx)
        Q_h=np.mean(Q_h_kx)

    if i==0:
        actual_flow_power_df = pd.DataFrame({'hour': t_datetime.hour, 'q_draw': np.mean(q_array_pde_act_kx),'control_step': t_datetime.hour * (int(3600 / mpc_dt)) + (t_datetime.minute * 60 / mpc_dt)},index=[pd.Timestamp(t_datetime)])
    else:
        actual_flow_power_df = pd.concat((actual_flow_power_df,pd.DataFrame({'hour': t_datetime.hour, 'q_draw': np.mean(q_array_pde_act_kx),'control_step': t_datetime.hour * (int(3600 / mpc_dt)) + (t_datetime.minute * 60 / mpc_dt)},index=[pd.Timestamp(t_datetime)])),axis=0)

    Q_array_cl[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt), 0] = Q_l_kx
    Q_array_cl[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt), 1] = Q_h_kx

    #log simulation outputs from PDE function
    if i==int(T/mpc_dt)-1:
        temp_sim_k[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt), :] = temp_sim_kx[0:int(mpc_dt/sim_dt),:]
        flowrate_mod[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt)] = flowrate_mod_kx[0:int(mpc_dt/sim_dt)]
        q_array_total_actual[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt)]=q_array_pde_act_kx[0:int(mpc_dt/sim_dt)]
        flowrate_mixed[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt)] = flowrate_m_kx[0:int(mpc_dt/sim_dt)]
    else:
        temp_sim_k[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt + 1), :] = temp_sim_kx
        flowrate_mod[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt)] = flowrate_mod_kx
        q_array_total_actual[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt)]=q_array_pde_act_kx
        flowrate_mixed[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt)] = flowrate_m_kx

    t_now = t_now + mpc_dt
    t_datetime=t_datetime+datetime.timedelta(seconds=mpc_dt)

    if model_type=='1nodemv':
        T0_f_last=T0_f
        T0_k_last = (T0_f_last - 32.0) * (5.0 / 9.0) + 273.15
        Q_l_last=Q_l
    if (model_type=='3nodemv') or (model_type=='3nodemvlf'):
        T0_l_f_last=T0_l_f
        T0_m_f_last=T0_m_f
        T0_h_f_last=T0_h_f
        T0_l_k_last = (T0_l_f_last - 32.0) * (5.0 / 9.0) + 273.15
        T0_m_k_last = (T0_m_f_last - 32.0) * (5.0 / 9.0) + 273.15
        T0_h_k_last = (T0_h_f_last - 32.0) * (5.0 / 9.0) + 273.15
        Q_l_last=Q_l
        Q_h_last=Q_h

######################################################################
##################        Parse results        #######################
######################################################################

#Unit conversion
temp_sim_F = (temp_sim_k - 273.15) * (9.0 / 5.0) + 32.0
#Sample full array at certain sensor locations
temp_sim_F_samp=temp_sim_F[:,sensor_ind]
temp_sim_k_samp=temp_sim_k[:,sensor_ind]

start_day_analysis=30.0 #calculate performance metrics from this day until the end of the simulation

#Electricity costs: (1) over whole period and (2) over last 26 days
cost_sim=np.dot(np.sum(Q_array_cl,axis=1),c_array_full)/((3600/sim_dt)*1000)
cost_sim2=np.dot(np.sum(Q_array_cl[int(start_day_analysis*T/(days_of_simulation*sim_dt)):,:],axis=1),c_array_full[int(start_day_analysis*T/(days_of_simulation*sim_dt)):])/((3600/sim_dt)*1000)

#Energy in drawn water (1) over whole period and (2) over last 26 days
flow_energy_sim2=np.sum(q_array_total_actual[int(start_day_analysis*T/(days_of_simulation*sim_dt)):])*(1/((60/sim_dt)*60*1000))
flow_energy_sim=np.sum(q_array_total_actual)*(1/((60/sim_dt)*60*1000))

flow_energy_target=np.sum(q_array_total[0:int(T/sim_dt)])*(1/((60/sim_dt)*60*1000))
flow_energy_target2=np.sum(q_array_total[int(start_day_analysis*T/(days_of_simulation*sim_dt)):int(T/sim_dt)])*(1/((60/sim_dt)*60*1000))

#Analyze water draw estimation accuracy
actual_flow_power=np.array(actual_flow_power_df.q_draw)
actual_flow_power_hr=np.mean(actual_flow_power.reshape(-1,int(3600/mpc_dt)),axis=1)/1000
est_flow_power_hr=np.mean(est_flow_power.reshape(-1,int(3600/mpc_dt)),axis=1)/1000
est_flow_power_hr=np.maximum(est_flow_power_hr,0)
rmse_flow_energy=(np.mean((actual_flow_power_hr-est_flow_power_hr)**2))**0.5
rmse_flow_energy2=(np.mean((actual_flow_power_hr[int(24.0*start_day_analysis):]-est_flow_power_hr[int(24.0*start_day_analysis):])**2))**0.5
actual_flow_power_hr_daily=np.mean(actual_flow_power_hr.reshape(int(days_of_simulation),24),axis=0)
est_flow_power_hr_daily=np.mean(est_flow_power_hr.reshape(int(days_of_simulation),24),axis=0)

#save flow estimates and actual values
os.chdir(out_dir)
np.savetxt('actflow_'+model_type+'_S'+sensor_config+'_'+forecast_type+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',actual_flow_power_hr)
np.savetxt('estflow_'+model_type+'_S'+sensor_config+'_'+forecast_type+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',est_flow_power_hr)

#Energy consumption: (1) over whole period and (2) over last 26 days
energy_heating_in_sim=((np.sum(Q_array_cl[:,0])/(3600/sim_dt))+(np.sum(Q_array_cl[:,1])/(3600/sim_dt)))/1000
energy_heating_in_sim2=((np.sum(Q_array_cl[int(start_day_analysis*T/(days_of_simulation*sim_dt)):,0])/(3600/sim_dt))+(np.sum(Q_array_cl[int(start_day_analysis*T/(days_of_simulation*sim_dt)):,1])/(3600/sim_dt)))/1000

#Average MPC computation time
comp_time_avg=np.mean(comp_time[0:int(start_day_analysis*T/(days_of_simulation*mpc_dt))])
comp_time_avg2=np.mean(comp_time[int(start_day_analysis*T/(days_of_simulation*mpc_dt)):])

#Cost per kWh of consumed energy, and per kWh of drawn water
cost_per_kwh_sim=cost_sim/energy_heating_in_sim
cost_per_flow_sim=cost_sim/flow_energy_sim
cost_per_kwh_sim2=cost_sim2/energy_heating_in_sim2
cost_per_flow_sim2=cost_sim2/flow_energy_sim2

#thermal comfort results. Bin draws by temperature
f_array_total2=flowrate_mixed[0:int(T/sim_dt)]
gal_out_less_90=np.sum(f_array_total2[(np.arange(f_array_total2.shape[0])>=int(start_day_analysis*T/(days_of_simulation*sim_dt)))&(temp_sim_F[:,M-1]<90)])*264.172*sim_dt
gal_out_90_95=np.sum(f_array_total2[(np.arange(f_array_total2.shape[0])>=int(start_day_analysis*T/(days_of_simulation*sim_dt)))&(temp_sim_F[:,M-1]<95)&(temp_sim_F[:,M-1]>=90)])*264.172*sim_dt
gal_out_95_100=np.sum(f_array_total2[(np.arange(f_array_total2.shape[0])>=int(start_day_analysis*T/(days_of_simulation*sim_dt)))&(temp_sim_F[:,M-1]<100)&(temp_sim_F[:,M-1]>=95)])*264.172*sim_dt
gal_out_100_105=np.sum(f_array_total2[(np.arange(f_array_total2.shape[0])>=int(start_day_analysis*T/(days_of_simulation*sim_dt)))&(temp_sim_F[:,M-1]<105)&(temp_sim_F[:,M-1]>=100)])*264.172*sim_dt
gal_out_105_110=np.sum(f_array_total2[(np.arange(f_array_total2.shape[0])>=int(start_day_analysis*T/(days_of_simulation*sim_dt)))&(temp_sim_F[:,M-1]<110)&(temp_sim_F[:,M-1]>=105)])*264.172*sim_dt
gal_out_110_115=np.sum(f_array_total2[(np.arange(f_array_total2.shape[0])>=int(start_day_analysis*T/(days_of_simulation*sim_dt)))&(temp_sim_F[:,M-1]<115)&(temp_sim_F[:,M-1]>=110)])*264.172*sim_dt
gal_out_115_120=np.sum(f_array_total2[(np.arange(f_array_total2.shape[0])>=int(start_day_analysis*T/(days_of_simulation*sim_dt)))&(temp_sim_F[:,M-1]<120)&(temp_sim_F[:,M-1]>=115)])*264.172*sim_dt
gal_out_more_120=np.sum(f_array_total2[(np.arange(f_array_total2.shape[0])>=int(start_day_analysis*T/(days_of_simulation*sim_dt)))&(temp_sim_F[:,M-1]>=120)])*264.172*sim_dt

results_temp=np.zeros((8,3))
results_temp[0,:]=np.array([0,90,gal_out_less_90])
results_temp[1,:]=np.array(([90,95,gal_out_90_95]))
results_temp[2,:]=np.array(([95,100,gal_out_95_100]))
results_temp[3,:]=np.array(([100,105,gal_out_100_105]))
results_temp[4,:]=np.array(([105,110,gal_out_105_110]))
results_temp[5,:]=np.array(([110,115,gal_out_110_115]))
results_temp[6,:]=np.array(([115,120,gal_out_115_120]))
results_temp[7,:]=np.array(([120,200,gal_out_more_120]))

#Performance metrics and other interesting statistics to be saved
results=np.zeros((8,2))
results[0,0]=energy_heating_in_sim
results[0,1]=energy_heating_in_sim2
results[1,0]=cost_sim
results[1,1]=cost_sim2
results[2,0]=cost_per_kwh_sim
results[2,1]=cost_per_kwh_sim2
results[3,0]=cost_per_flow_sim
results[3,1]=cost_per_flow_sim2
results[4,0]=flow_energy_sim
results[4,1]=flow_energy_sim2
results[5,0]=comp_time_avg
results[5,1]=comp_time_avg2
results[6,0]=rmse_flow_energy
results[6,1]=rmse_flow_energy2
results[7,0]=flow_energy_target
results[7,1]=flow_energy_target2

#save results
os.chdir(out_dir)
if (forecast_type == 'hist_avg') or (forecast_type == 'hist_avg_fest'):
    np.savetxt('results_'+model_type+'_S'+sensor_config+'_P'+forecast_type+'_N'+str(n_samp)+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',results)
    np.savetxt('outtemp_'+model_type+'_S'+sensor_config+'_P'+forecast_type+'_N'+str(n_samp)+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',results_temp)
if (forecast_type == 'hist_quantile') or (forecast_type == 'hist_quantile_fest'):
    np.savetxt('results_'+model_type+'_S'+sensor_config+'_P'+forecast_type+'_N'+str(n_samp)+'_Q'+str(int(quant*100))+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',results)
    np.savetxt('outtemp_'+model_type+'_S'+sensor_config+'_P'+forecast_type+'_N'+str(n_samp)+'_Q'+str(int(quant*100))+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',results_temp)
if forecast_type=='perfect':
    np.savetxt('results_'+model_type+'_S'+sensor_config+'_P'+forecast_type+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',results)
    np.savetxt('outtemp_'+model_type+'_S'+sensor_config+'_P'+forecast_type+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',results_temp)

#Plot timeseries
if save_fig==True:

    plt.figure(figsize=(10,6))
    plt.subplot(4,1,1)
    plt.plot(np.arange(Q_array_cl.shape[0])/(3600/sim_dt),Q_array_cl[:,0],label='Lower')
    plt.plot(np.arange(Q_array_cl.shape[0])/(3600/sim_dt),Q_array_cl[:,1],label='Upper')
    plt.legend(fontsize=14)
    plt.grid()
    plt.xlim([24*(32),24*(35)])
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.ylabel('Power (W)',fontsize=14)

    plt.subplot(4,1,2)
    for i in range(temp_sim_F_samp.shape[1]):
        plt.plot(np.arange(temp_sim_F_samp.shape[0])/(3600/sim_dt),temp_sim_F_samp[:,i])
    plt.grid()
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlim([24*(32),24*(35)])
    plt.ylabel('Temperature (F)',fontsize=14)
    ax.set_yticks([80,100,120,140,160])
    plt.ylim([65,170])

    plt.subplot(4,1,3)
    plt.plot(np.arange(len(c_array_full))/(3600/sim_dt),c_array_full,label='TOU ELEC')
    plt.grid()
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlim([24*(32),24*(35)])
    plt.ylabel('Cost ($)',fontsize=14)
    plt.ylim([0,0.6])
    plt.legend(fontsize=14)
    plt.xlabel('Hour', fontsize=14)

    plt.subplot(4,1,4)
    plt.plot(np.arange(len(f_array_total2))/(3600/sim_dt),f_array_total2)
    plt.grid()
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlim([24*(32),24*(35)])
    plt.ylabel('Cost ($)',fontsize=14)
    #plt.ylim([0,0.6])
    plt.legend(fontsize=14)
    plt.xlabel('Hour', fontsize=14)

    plt.tight_layout()
    os.chdir(out_dir)
    if  (forecast_type=='hist_avg') or (forecast_type=='hist_avg_fest'):
        plt.savefig('plot_'+model_type+'_S'+sensor_config+'_'+forecast_type+'_N'+str(n_samp)+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.png',dpi=300)
    if (forecast_type=='hist_quantile') or (forecast_type=='hist_quantile_fest'):
        plt.savefig('plot_'+model_type+'_S'+sensor_config+'_'+forecast_type+'_N'+str(n_samp)+'_Q'+str(int(quant*100))+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.png',dpi=300)
    if (forecast_type=='perfect'):
        plt.savefig('plot_'+model_type+'_S'+sensor_config+'_'+forecast_type+'_H'+str(horizon_hr)+'_'+flow_type.replace('_','')+'_'+cost_name+'.png',dpi=300)
