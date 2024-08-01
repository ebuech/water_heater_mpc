#%%

import os
cwd=os.getcwd()
import sys
sys.path.append(os.path.join(cwd, 'functions'))
import ast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import thermostat
import multinode_sim

######################################################################
##############        Define scenario parameters        ##############
######################################################################


T=3600*24*(28+28) # Simulation horizon [seconds]
days_of_simulation=T/(3600*24)
mpc_dt=30 #frequency of thermostat controls [s]
sim_dt=10 #multinode simulation timestep [s]
ritchie_num=72 #house in ritchie flow dataset. Homes used in paper include: 0,18,20,37,42,67,63,72
flow_type='ritchie_'+str(int(ritchie_num))
save_fig=False #Turn figure saving on/off
cost_name='TPDP' #Electricity cost profile ['elec' for PG&E TOU, 'TPDP' for CalFlexHub dynamic rate]
f_multinode_params='multinode.txt'

model_dir=os.path.join(cwd, 'data/model_parameters')
out_dir=os.path.join(cwd,'results')

#Controller parameters
f_model_params='model_thermostat_mv_240V.txt'

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
f_array_total=np.ndarray.flatten(np.tile(f_array_test_min.reshape(-1,1),(1,int(60/sim_dt))).reshape(-1,1))[0:int(T/sim_dt)]



######################################################################
###########        load electricity cost profile        ##############
######################################################################

if cost_name=='elec': #PG&E TOU profile
    cost_filename='pge_'+cost_name+'_60day.csv'
if cost_name=='TPDP': #CalFlexHub HDP profile
    cost_filename='MIDAS_rate_60day_USCA-CFCF-'+cost_name+'-0000.csv'

#load cost profile
os.chdir(os.path.join(cwd, 'data/electricity_cost'))
cost_sch = pd.read_csv(cost_filename)
c_array_total = np.zeros((int(T/sim_dt),))
for i in range(cost_sch.shape[0]):
    c_array_total[int(cost_sch.time[i] / sim_dt):int((cost_sch.time[i] / sim_dt) + (cost_sch.duration[i] / sim_dt))] = float(cost_sch.cost[i])


######################################################################
###########    Define multi-node model parameters     ################
######################################################################

os.chdir(model_dir)
f=open(f_multinode_params,'r')
wh_params=f.read()
f.close()
wh_params=ast.literal_eval(wh_params)

#sensor heights in tank
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

r=wh_params['r_in']/39.37 #[m] tank radius
dx=h/wh_params['M'] #[m] node height
A_surf=dx*np.pi*r*2 #[m^2] surface area
slug_cross_area=np.pi*(r**2) # cross sectional area
slug_vol=dx*slug_cross_area #[m^3] node volume
cap=slug_vol*wh_params['cp']*wh_params['rho'] #node thermal capacitance

#initial profile (starting around 120 deg F for most of tank)
temp_sim_k = np.zeros((int(T/sim_dt),wh_params['M']))
temp_sim_k[0,:]=np.array(wh_params['T_init_prof'])

#put all parameters into dictionary
#wh_params = {}
#wh_params['M'] = M
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

#target draw power trajectories
q_array_total=f_array_total*wh_params['rho']*wh_params['cp']*(mv_set_k-T_in)
data_full_10min['q_draw']=data_full_10min.m3s*wh_params['rho']*wh_params['cp']*(mv_set_k-T_in)


######################################################################
###################      Control simulation       ####################
######################################################################


state = 0 #thermostatic control state
t_now = 0
init_bool = False
prev_sol = {}

Q_array_cl = np.zeros((int(T/sim_dt), 2)) #array for logging power consumption
flowrate_mod=np.zeros((int(T/sim_dt),)) #Array for logging outlet flow rate
flowrate_mixed=np.zeros((int(T/sim_dt),)) #Array for logging mixed flow rate
q_array_total_actual=np.zeros((int(T/sim_dt),)) #Array for logging actual draw power values

#step through control timesteps
for i in range(int(T / mpc_dt)):
    t1 = time.time()

    #current temperature
    temp_now = np.array(temp_sim_k[int(t_now/sim_dt), :])
    temp_now_samp = temp_now[sensor_ind]
    temp_now_samp_F = (temp_now_samp - 273.15) * (9.0 / 5.0) + 32.0

    #get temperature sensor values
    T0_m_f = temp_now_samp_F[6]
    T0_h_f = temp_now_samp_F[7]

    #thermostatic control
    ul_opt, uh_opt, state = thermostat.thermostat(T0_m_f, T0_h_f, state,f_model_params,model_dir)
    Q_l = ul_opt
    Q_h = uh_opt
    Q_array_cl[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt + 1), 0] = Q_l
    Q_array_cl[int(i * mpc_dt/sim_dt):int((i + 1) * mpc_dt/sim_dt + 1), 1] = Q_h

    q_array_pde = (q_array_total[int(i * mpc_dt / sim_dt):int((i + 1) * mpc_dt / sim_dt)])

    #simulate multinode model
    temp_sim_kx,amb_loss_kx,flowrate_mod_kx,q_array_pde_act_kx,flowrate_m_kx = multinode_sim.sim_multinode_thermostat(wh_params, sim_dt, mpc_dt, Q_l, Q_h, q_array_pde, temp_now)

    #put simulation outputs from pde function into an array
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

######################################################################
##################        Parse results        #######################
######################################################################

#convert temperature results from K to F, and select out sensor locations
temp_sim_F = (temp_sim_k - 273.15) * (9.0 / 5.0) + 32.0
temp_sim_F_samp=temp_sim_F[:,sensor_ind]
temp_sim_k_samp=temp_sim_k[:,sensor_ind]

start_day_analysis=30.0 #calculate performance metrics from this day until the end of the simulation

#Electricity costs: (1) over whole period and (2) over last 26 days
cost_sim=np.dot((np.sum(Q_array_cl,axis=1)),c_array_total)/(3600/sim_dt*1000)
cost_sim2=np.dot((np.sum(Q_array_cl[int(start_day_analysis*T/(days_of_simulation*sim_dt)):,:],axis=1)),c_array_total[int(start_day_analysis*T/(days_of_simulation*sim_dt)):])/((3600/sim_dt)*1000)

#Energy in drawn water (1) over whole period and (2) over last 26 days
flow_energy_sim2=np.sum(q_array_total_actual[int(start_day_analysis*T/(days_of_simulation*sim_dt)):])*(1/((60/sim_dt)*60*1000))
flow_energy_sim=np.sum(q_array_total_actual)*(1/((60/sim_dt)*60*1000))

flow_energy_target=np.sum(q_array_total[0:int(T/sim_dt)])*(1/((60/sim_dt)*60*1000))
flow_energy_target2=np.sum(q_array_total[int(start_day_analysis*T/(days_of_simulation*sim_dt)):int(T/sim_dt)])*(1/((60/sim_dt)*60*1000))

#Energy consumption: (1) over whole period and (2) over last 26 days
energy_heating_in_sim=((np.sum(Q_array_cl[:,0])/(3600/sim_dt))+(np.sum(Q_array_cl[:,1])/(3600/sim_dt)))/1000
energy_heating_in_sim2=((np.sum(Q_array_cl[int(start_day_analysis*T/(days_of_simulation*sim_dt)):,0])/(3600/sim_dt))+(np.sum(Q_array_cl[int(start_day_analysis*T/(days_of_simulation*sim_dt)):,1])/(3600/sim_dt)))/1000

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
results=np.zeros((6,2))
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
results[5,0]=flow_energy_target
results[5,1]=flow_energy_target2

#save results
os.chdir(out_dir)
np.savetxt('results_thermostat'+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',results)
np.savetxt('outtemp_thermostat_'+'_'+flow_type.replace('_','')+'_'+cost_name+'.txt',results_temp)


#Plot timeseries
if save_fig==True:

    plt.figure(figsize=(10,6))
    plt.subplot(3,1,1)
    plt.plot(np.arange(Q_array_cl.shape[0])/(3600/sim_dt),Q_array_cl[:,0],label='Lower')
    plt.plot(np.arange(Q_array_cl.shape[0])/(3600/sim_dt),Q_array_cl[:,1],label='Upper')
    plt.legend(fontsize=14)
    plt.grid()
    plt.xlim([24*(32),24*(35)])
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.ylabel('Power (W)',fontsize=14)

    plt.subplot(3,1,2)
    for i in range(temp_sim_F_samp.shape[1]):
        plt.plot(np.arange(temp_sim_F_samp.shape[0])/(3600/sim_dt),temp_sim_F_samp[:,i])
    plt.grid()
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlim([24*(32),24*(35)])
    plt.ylabel('Temperature (F)',fontsize=14)
    ax.set_yticks([80,100,120,140,160])
    plt.ylim([65,160])

    plt.subplot(3,1,3)
    plt.plot(np.arange(len(c_array_total))/(3600/sim_dt),c_array_total,label='TOU ELEC')
    plt.grid()
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlim([24*(32),24*(35)])
    plt.ylabel('Cost ($)',fontsize=14)
    plt.ylim([0,0.6])
    plt.legend(fontsize=14)
    plt.xlabel('Hour', fontsize=14)

    plt.tight_layout()
    os.chdir(out_dir)
    plt.savefig('plot_thermostat_'+flow_type.replace('_','')+'_'+cost_name+'.png',dpi=300)
