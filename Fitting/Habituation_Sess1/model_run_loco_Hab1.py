# Imports

import sys
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from lmfit import minimize, Parameter, Parameters, report_fit
from model_functions_loco_Hab1 import *

########## DATA LOAD ##########################################################

# timepoint
dt = 0.164745

# PC_average (whole population)
pc_all = np.loadtxt(r'QA_Nor_PC_All_AvgSess1.txt')
# PC_average_standard_error
pc_all_err = np.loadtxt(r'QA_Nor_PC_All_SEMSess1.txt')

# Experimental time calculation
t_exp_1 = exp_time(start=0.0, step=dt, count=len(pc_all))

# PV_average (whole population)
pv = np.loadtxt(r'QA_Nor_PV_AvgSess1.txt')
# PV_average_standard_error
pv_err = np.loadtxt(r'QA_Nor_PV_SEMSess1.txt')

# SST_average (whole population)
sst = np.loadtxt(r'QA_Nor_SST_AvgSess1.txt')
# SST_average_standard_error
sst_err = np.loadtxt(r'QA_Nor_SST_SEMSess1.txt')

# VIP_average (positives and negatives)
vip_pos = np.loadtxt(r'QA_Nor_VIP_AvgSess1_Pos.txt')
vip_neg = np.loadtxt(r'QA_Nor_VIP_AvgSess1_Neg.txt')

# VIP_average_standard_error
vip_pos_err = np.loadtxt(r'QA_Nor_VIP_SEMSess1_Pos.txt')
vip_neg_err = np.loadtxt(r'QA_Nor_VIP_SEMSess1_Neg.txt')
input_file = "sobol_samples_Hab1reducedYehor.csv"

if os.path.exists(input_file):
    df_existing = pd.read_csv(input_file)
    last_id = df_existing["ID"].max()
else:
    print("Sobol sample file didn't exist. Created a new one.")
    df_existing = pd.DataFrame()
    last_id = 0


###############################################################################

########## TRACE CUTTING ######################################################

# Inicialization of new arrays to cut traces for suitable range

t_exp_1_new = []
pv_new = []
sst_new = []
vip_pos_new = []
vip_neg_new = []
pc_all_new = []
pc_all_err_new = []
pv_err_new = []
sst_err_new = []
vip_pos_err_new = []
vip_neg_err_new = []

# Cutting the traces
for ind, time in np.ndenumerate(t_exp_1):
    if (time >= 5) and (time <= 25):
        t_exp_1_new.append(time - 5)
        pv_new.append(pv[ind])
        sst_new.append(sst[ind])
        vip_pos_new.append(vip_pos[ind])
        vip_neg_new.append(vip_neg[ind])
        pc_all_new.append(pc_all[ind])
        pv_err_new.append(pv_err[ind])
        sst_err_new.append(sst_err[ind])
        vip_pos_err_new.append(vip_pos_err[ind])
        vip_neg_err_new.append(vip_neg_err[ind])
        pc_all_err_new.append(pc_all_err[ind])

# Adjusting for a missing initial point after deconvolution
pv_new.insert(0, pv_new[0])
sst_new.insert(0, sst_new[0])
vip_pos_new.insert(0, vip_pos_new[0])
vip_neg_new.insert(0, vip_neg_new[0])
pc_all_new.insert(0, pc_all_new[0])
pv_err_new.insert(0, pv_err_new[0])
sst_err_new.insert(0, sst_err_new[0])
vip_pos_err_new.insert(0, vip_pos_err_new[0])
vip_neg_err_new.insert(0, vip_neg_err_new[0])
pc_all_err_new.insert(0, pc_all_err_new[0])

t_exp_1_new.append(t_exp_1_new[-1] + dt)
###############################################################################

########## VARIABLES AND CONSTANTS ############################################

t = np.float32(np.array(t_exp_1_new))
t_exp = t

param_val = float(os.environ.get("PARAM_VAL", 1)) # Row id for getting parameters from dataframe (df_existing). 1 - default value to set if PARAM_VAL is not set.
batch_id = str(os.environ.get("BATCH_ID", 1))
print("Test param_val", param_val)
row = df_existing[df_existing['ID'] == param_val].iloc[0]
weights = row.drop("ID").values

threshold = np.float32(0)

power = np.float32(2.0)
q = np.float32(1.0)

ampl_1 = 1.0

r_1 = np.float32(1.713939)
delay_1 = np.float32(0.1650418)
delay_2 = np.float32(1.73305)
delay_3 = np.float32(0.3695274)
decay = np.float32(3.324089)
decay_s = np.float32(0.2685497)
decay_f = np.float32(0.8342502)
decay_ff = np.float32(2.152722)
ampl = np.float32(1.0)
base = np.float32(1.0)
base_sigm = np.float32(0)
s_start = np.float32(0.1835884)
k = np.float32(weights[24])

init = np.float32(np.array([0.055, 0.011, 0.017, 0.208, 1.000]))
tau = np.float32(np.array([0.015, 0.0075, 0.019, 0.019]))
i = np.float32(np.array([0.28618393, 0.00729576, 0.09371017, 0.51220849, 0.94359329]))

w = np.float32(np.array([
              weights[0],  # w_0  PC  -> PC
              weights[1],  # w_1  FF  -> PC
              weights[2],  # w_2  SM   -> PC
              weights[3],  # w_3  PV -> PC
              weights[4],  # w_4  SST  -> PC
              weights[5],  # w_5  PC   -> PV
              weights[6],  # w_6  FF  -> PV
              weights[7],  # w_7  SM -> PV
              weights[8],  # w_8  PV   -> PV
              weights[9],  # w_9  SST  -> PV
              weights[10],  # w_10 PC -> SST
              weights[11],  # w_11 FB  -> SST
              weights[12],  # w_12 VIP_P  -> SST
              weights[13],  # w_13 PC  -> VIP_P
              weights[14],  # w_14 SST  -> VIP_P
              weights[15],  # w_15 SM  -> VIP_P
              weights[16],  # w_16 FB  -> PC
              weights[17],  # w_17 FB  -> PV
              weights[18],  # w_18 FB -> VIP_P
              weights[19],  # w_19 FB  -> VIP_N
              weights[20],  # w_20 PC  -> VIP_N
              weights[21],  # w_21 SST  -> VIP_N
              weights[22],  # w_22 SM -> VIP_N
              weights[23]]))  # w_23 VIP_N -> SST


data_pv = np.float32(np.array(pv_new))
data_sst = np.float32(np.array(sst_new))
data_vip_pos = np.float32(np.array(vip_pos_new))
data_vip_neg = np.float32(np.array(vip_neg_new))
data_pc_all = np.float32(np.array(pc_all_new))

pc_all_err_new = np.float32(np.array(pc_all_err_new))
pv_err_new = np.float32(np.array(pv_err_new))
sst_err_new = np.float32(np.array(sst_err_new))
vip_pos_err_new = np.array(vip_pos_err_new)
vip_neg_err_new = np.array(vip_neg_err_new)
###############################################################################

########## SETTING PARAMETERS #################################################

"""
Setting lmfit.Parameters() object with all parameters for the model,
their initial values, ranges and boolean varaible wether to vary them or not during fitting
"""

params = Parameters()
params.add('w_0', value = w[0] , vary = True, min = 0.0, max = 1.0)
params.add('w_1', value = w[1] , vary = True, min = 0.0, max = 4)
params.add('w_2', value = w[2] , vary = True, min = 0.0, max = 4)
params.add('w_3', value = w[3] , vary = True, min = 0.0, max = 4)
params.add('w_4', value = w[4] , vary = True, min = 0.0, max = 4)
params.add('w_5', value = w[5] , vary = True, min = 0.0, max = 4)
params.add('w_6', value = w[6] , vary = True, min = 0.0, max = 4)
params.add('w_7', value = w[7] , vary = True, min = 0.0, max = 4)
params.add('w_8', value = w[8] , vary = True, min = 0.0, max = 4)
params.add('w_9', value = w[9] , vary = True, min = 0.0, max = 4)
params.add('w_10', value = w[10] , vary = True, min = 0.0, max = 4)
params.add('w_11', value = w[11] , vary = True, min = 0.0, max = 4)
params.add('w_12', value = w[12] , vary = True, min = 0.0, max = 4)
params.add('w_13', value = w[13] , vary = True, min = 0.0, max = 4)
params.add('w_14', value = w[14] , vary = True, min = 0.0, max = 4)
params.add('w_15', value = w[15] , vary = True, min = 0.0, max = 4)
params.add('w_16', value = w[16] , vary = True, min = 0.0, max = 4)
params.add('w_17', value = w[17] , vary = True, min = 0.0, max = 4)
params.add('w_18', value = w[18] , vary = True, min = 0.0, max = 4)
params.add('w_19', value = w[19] , vary = True, min = 0.0, max = 4)
params.add('w_20', value = w[20] , vary = True, min = 0.0, max = 4)
params.add('w_21', value = w[21] , vary = True, min = 0.0, max = 4)
params.add('w_22', value = w[22] , vary = True, min = 0.0, max = 4)
params.add('w_23', value = w[23] , vary = True, min = 0.0, max = 4)

params.add('tau_0', value = tau[0] , vary = False, min = 0.001, max = 0.03)
params.add('tau_1', value = tau[1] , vary = False, min = 0.001, max = 0.03)
params.add('tau_2', value = tau[2] , vary = False, min = 0.001, max = 0.03)
params.add('tau_3', value = tau[3] , vary = False, min = 0.001, max = 0.03)
params.add('threshold', value = threshold , vary = False, min = -np.inf, max = np.inf)
params.add('power', value = power , vary = False, min = 0.5, max = 2.5)
params.add('q', value = q, vary = False, min = 0.001, max = 2.5 )
params.add('i_0', value = i[0] , vary = True, min = 0.0, max = 0.7)
params.add('i_1', value = i[1] , vary = True, min = 0.0, max = 0.7)
params.add('i_2', value = i[2] , vary = True, min = 0.0, max = 0.7)
params.add('i_3', value = i[3] , vary = True, min = 0.0, max = 0.7)
params.add('i_4', value = i[4] , vary = True, min = 0.0, max = 2.7)

params.add('ampl_1', value = ampl_1 , vary = False, min = 0.8, max = 1.5)
params.add('r_1', value = r_1, vary = False, min = 0, max = 5)
params.add('delay_1', value = delay_1, vary = False, min = 0, max = 1.5)
params.add('delay_2', value = delay_2, vary = False, min = 0, max = 25)
params.add('delay_3', value = delay_3, vary = False, min = 0, max = 3)
params.add('decay', value = decay, vary = False, min = 0, max = 10)
params.add('decay_s', value = decay_s, vary = True, min = 0, max = 10)
params.add('decay_f', value = decay_f, vary = True, min = 0, max = 10)
params.add('decay_ff', value = decay_ff, vary = True, min = 0, max = 10)
params.add('ampl', value = ampl, vary = False, min = 0, max = 5)
params.add('base', value = base, vary = False, min = 0, max = 5)
params.add('base_sigm', value = base_sigm, vary = False, min = 0, max = 2)
params.add('s_start', value = s_start, vary = False, min = 0, max = 4)
params.add('k', value = k, vary = True, min = 0.005, max = 0.15)
###############################################################################

########## FITTING NELDER #####################################################

result_nelder = minimize(residual_step, params, method='nelder',
                         args=(t_exp, init, data_pc_all, data_pv, data_sst, data_vip_pos, data_vip_neg,
                               pc_all_err_new, pv_err_new, sst_err_new, vip_pos_err_new, vip_neg_err_new),
                         nan_policy='propagate', options={'adaptive':True},
                         max_nfev=1250)

RMSE_full_nelder = RMSE_full(result_nelder.params,
                             init, data_pc_all, data_pv, data_sst, data_vip_pos, data_vip_neg, t_exp)

RMSE_full_nelder_stim = RMSE_full_stim(result_nelder.params,
                             init, data_pc_all, data_pv, data_sst, data_vip_pos, data_vip_neg, t_exp)


# TODO: Write result in the output file.
rows = []

for name, param in params.items():
    rows.append(
        {"ID": param_val,
         "Parameter": name+"_init",
         "Value": f"{param.value:11.5f}",
         "Stderr": "N/A"}
    )

for name, param in result_nelder.params.items():
    stderr_val = f"{param.stderr:11.5f}" if param.stderr is not None else "N/A"
    rows.append(
        {"ID": param_val,
        "Parameter": name,
        "Value": f"{param.value:11.5f}",
        "Stderr": stderr_val}
    )

rows.append(
        {"ID": param_val,
        "Parameter": "chisqr",
        "Value": f"{result_nelder.chisqr}",
        "Stderr": "N/A"}
    )
rows.append(
        {"ID": param_val,
        "Parameter": "redchi",
        "Value": f"{result_nelder.redchi}",
        "Stderr": "N/A"}
    )

rows.append({
    "ID": param_val,
    "Parameter": "RMSE_full",
    "Value": f"{RMSE_full_nelder}",
    "Stderr": "N/A"
})
rows.append({
    "ID": param_val,
    "Parameter": "RMSE_full_stim",
    "Value": f"{RMSE_full_nelder_stim}",
    "Stderr": "N/A"
})

df_result = pd.DataFrame(rows)
df_result.to_csv(f"results/batch_{batch_id}/output_ID_{int(param_val)}.csv", index=False)

print(report_fit(result_nelder))
print(RMSE_full_nelder)
print(RMSE_full_nelder_stim)
###############################################################################
