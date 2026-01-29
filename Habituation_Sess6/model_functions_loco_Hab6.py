# Imports

import sys
import pandas as pd
import numpy as np
import random
from scipy.integrate import odeint, solve_ivp
from lmfit import minimize, Parameter, Parameters, report_fit


def Step(t, stim=5, inter=10, amp=1.0, base=0, k=0, delay=0, decay_f=0):
    """
    Feedback (FB) input:
    ------------------------------------
    Represented as a step function. Returns value of the input function in a time point t.
    ------------------------------------
    Parameters:

    t (running)             - time in seconds;
    stim (fixed) = 5 s      - time of the stimulus start;
    inter (fixed) = 10 s    - duration of the stimulus;
    amp (fixed) = 1.0 Hz    - amplitude of response;
    base (fixed) = 0 Hz     - baseline activity;
    k (fixed) = 0           - slope of the slow component. Used to be varied while testing linear depression or sensitization component in the FB input;
    delay (variable)        - delay of the FB input to the cell relative to the stimulus start;

    """

    if (t < stim + delay):
        h = base

    elif (t > stim + inter):
        h = amp * np.exp(-(t - stim - inter) * decay_f)

    else:
        h = amp * (t - stim - delay) * k * 0.164745 + amp

    return h

# In Sigm() dt=0 now, apparently, it was 1 timepoint in original fit.
# However, this works fine.
def Sigm(t, stim=5, inter=10, ampl=1.0, base=0, rate=1, delay=0, decay_s=1, dt=0):
    """
    Slow modulation (SM) input:
    ------------------------------------
    Represented as a sigmoid function. Returns value of the input function in a time point t.
    ------------------------------------
    Parameters:

    t (running)               - time in seconds;
    stim (fixed) = 5 s        - time of the stimulus start;
    inter (fixed) = 10 s      - duration of the stimulus;
    ampl (fixed) = 1.0 Hz     - amplitude of response;
    base (fixed) = 0 Hz       - baseline activity;
    rate (variable)           - time-constant of the SM input;
    delay (variable)          - shift of the sigmoid center relative to stimulus start;
    decay_s (varaible)        - time-constant of the SM input exponential decay after the end of stimulation;

    """

    if (t < stim):
        h = base

    elif (t > stim + inter):
        h = ((base + (ampl / (1 + np.exp((delay - inter - dt) / rate))))
             * np.exp(-(t - stim - inter) * decay_s) + base)

    else:
        h = base + (ampl / (1 + np.exp((stim + delay - t) / rate)))

    return h


def expon(t, stim=5, inter=10, ampl=1.5, base=0, decay=1, delay=0, b=0, decay_ff=0, s_start=0.1, k=0.1):
    """
    Feedforward (FF) input:
    ------------------------------------
    Represented as a flat step function with fast exponential decay on the stimulus start and linear increase during 10 second period.
    Returns value of the input function in a time point t.
    ------------------------------------
    Parameters:

    t (running)               - time in seconds;
    stim (fixed) = 5 s        - time of the stimulus start;
    inter (fixed) = 10 s      - duration of the stimulus;
    ampl (variable)           - amplitude of peak;
    base (fixed) = 1 Hz       - steady-state firing rate after fast exponential depression;
    decay (variable)          - time-constant of the fast exponential depression;
    delay (variable)          - delay of the FF input to the cell relative to the stimulus start;
    b (fixed) = 0 Hz          - baseline activity;
    decay_ff (varaible)       - time-constant of the FF input exponential decay after the end of stimulation;
    s_start (varaible)        - delay after stimulus when linear modulation starts;
    k (varaible)              - slope of the slow linear modulation;

    """

    if (t < stim + delay):
        h = b

    elif (t > stim + inter):
        h = (b + base + ampl * np.exp(-(inter - delay) * decay)
             + (inter - delay - s_start) * k) * np.exp(-(t - stim - inter) * decay_ff)

    elif ((t >= stim + delay) and (t < stim + delay + s_start)):
        h = b + base + ampl * np.exp(-(t - stim - delay) * decay)

    else:
        h = (b + base + ampl * np.exp(-(t - stim - delay) * decay) + ((t - stim - delay - s_start) * k))

    return h


def model_step(t,
               y,
               w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, #Neg
               tau_0, tau_1, tau_2, tau_3,
               threshold,
               power,
               q,
               i_0, i_1, i_2, i_3, i_4, #Neg
               r_1, decay, delay_1, delay_2, delay_3, ampl, base, decay_s, ampl_1, base_sigm, decay_f, decay_ff,
               s_start, k
               ):
    """
    Model basis:
    ------------------------------------
    Systems of first order differential equations that captures activity of populations without (df_xdt) and with (df_x_odt) optogenetic manipulations.
    Returns an array of values of the derrivatives at a certain timepoint, calculated from values of functions in previous timepoint.
    ------------------------------------
    Parameters:

    t (running)               - time in seconds;
    y (running)               - values of activities on the previous timepoint;
    w_x (variable)            - synaptic weights;
    tau_x (fixed)             - neurons time-constants;
    threshold (fixed) = 0 Hz  - minimum rectification value;
    power (fixed) = 2         - value of the power in the input-output function;
    q (fixed) = 1             - normalization coeficient in the input-output function;
    i_x (variable)            - baseline activity of neurons;
    ...

    """

    f_e, f_p, f_s, f_v_pos, f_v_neg = y #

    ff_e = (min(max((i_0
                     + w_0 * f_e
                     + w_1 * expon(t, ampl=ampl, base=base, decay=decay,
                                   delay=delay_1, decay_ff=decay_ff, s_start=s_start, k=k)
                     + w_2 * Sigm(t, ampl=ampl_1, rate=r_1, delay=delay_2,
                                  decay_s=decay_s, base=base_sigm)
                     + w_16 * Step(t, k=0, delay=delay_3, decay_f=decay_f)
                     - w_3 * f_p
                     - w_4 * f_s), threshold), 25))

    ff_p = (min(max((i_1
                     + w_17 * Step(t, k=0, delay=delay_3, decay_f=decay_f)
                     + w_5 * f_e
                     + w_6 * expon(t, ampl=ampl, base=base, decay=decay,
                                   delay=delay_1, decay_ff=decay_ff, s_start=s_start, k=k)
                     + w_7 * Sigm(t, ampl=ampl_1, rate=r_1, delay=delay_2,
                                  decay_s=decay_s, base=base_sigm)
                     - w_8 * f_p
                     - w_9 * f_s), threshold), 25))
    ff_s = (min(max((i_2
                     + w_10 * f_e
                     + w_11 * Step(t, k=0, delay=delay_3, decay_f=decay_f)
                     - w_12 * f_v_pos
		     - w_23 * f_v_neg
                     ), threshold), 25)) #

    ff_v_pos = (min(max((i_3
                     + w_18 * Step(t, k=0, delay=delay_3, decay_f=decay_f)
                     + w_13 * f_e
                     - w_14 * f_s
                     + w_15 * Sigm(t, ampl=ampl_1, rate=r_1, delay=delay_2,
                                   decay_s=decay_s, base=base_sigm)),threshold), 25))
    
    ff_v_neg = (min(max((i_4 
                    + w_19 * Step(t, k = 0, delay = delay_3, decay_f = decay_f) 
                    + w_20 * f_e 
                    - w_21 * f_s 
                    +  w_22 * Sigm(t, ampl = ampl_1, rate = r_1, delay = delay_2, 
                                   decay_s = decay_s, base = base_sigm)), threshold), 25))


    df_edt = ((q * ff_e ** power) - f_e) / tau_0
    df_pdt = ((q * ff_p ** power) - f_p) / tau_1
    df_sdt = ((q * ff_s ** power) - f_s) / tau_2
    df_v_posdt = ((q * ff_v_pos ** power) - f_v_pos) / tau_3
    df_v_negdt = ((q * ff_v_neg ** power) - f_v_neg) / tau_3

    dydt = [df_edt, df_pdt, df_sdt, df_v_posdt, df_v_negdt] # 

    return dydt


def exp_time(start, step, count, endpoint=False):
    """
    Experimental timepoints calculation:
    ------------------------------------
    Returns an array of values of the experimental timepoints.
    ------------------------------------
    Parameters:

    start              - starting point;
    step               - value of time step of experimental recordings;
    count              - number of points;

    """
    stop = start + (step * count)
    return np.linspace(start, stop, count, endpoint=endpoint)


def odesol_step(tt, init, params):
    """
    Solves differential equation system defined in model_step() function.
    """
    y_init = init
    w_0 = params['w_0'].value
    w_1 = params['w_1'].value
    w_2 = params['w_2'].value
    w_3 = params['w_3'].value
    w_4 = params['w_4'].value
    w_5 = params['w_5'].value
    w_6 = params['w_6'].value
    w_7 = params['w_7'].value
    w_8 = params['w_8'].value
    w_9 = params['w_9'].value
    w_10 = params['w_10'].value
    w_11 = params['w_11'].value
    w_12 = params['w_12'].value
    w_13 = params['w_13'].value
    w_14 = params['w_14'].value
    w_15 = params['w_15'].value
    w_16 = params['w_16'].value
    w_17 = params['w_17'].value
    w_18 = params['w_18'].value
    w_19 = params['w_19'].value #Neg 
    w_20 = params['w_20'].value #Neg 
    w_21 = params['w_21'].value #Neg 
    w_22 = params['w_22'].value #Neg 
    w_23 = params['w_23'].value #Neg 

    tau_0 = params['tau_0'].value
    tau_1 = params['tau_1'].value
    tau_2 = params['tau_2'].value
    tau_3 = params['tau_3'].value
    threshold = params['threshold'].value
    power = params['power'].value
    q = params['q'].value
    i_0 = params['i_0'].value
    i_1 = params['i_1'].value
    i_2 = params['i_2'].value
    i_3 = params['i_3'].value
    i_4 = params['i_4'].value #Neg
    ampl_1 = params['ampl_1'].value
    r_1 = params['r_1'].value
    delay_1 = params['delay_1'].value
    delay_2 = params['delay_2'].value
    delay_3 = params['delay_3'].value
    decay = params['decay'].value
    decay_s = params['decay_s'].value
    decay_f = params['decay_f'].value
    decay_ff = params['decay_ff'].value
    ampl = params['ampl'].value
    base = params['base'].value
    base_sigm = params['base_sigm'].value
    s_start = params['s_start'].value
    k = params['k'].value

    sol = solve_ivp(lambda t, y: model_step(t, y,
                                            w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13,
                                            w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, #Neg
                                            tau_0, tau_1, tau_2, tau_3,
                                            threshold,
                                            power,
                                            q,
                                            i_0, i_1, i_2, i_3, i_4, #Neg
                                            r_1, decay, delay_1, delay_2, delay_3, ampl, base, decay_s, ampl_1,
                                            base_sigm, decay_f, decay_ff, s_start, k
                                            ), 
                    [tt[0], tt[-1]],
                    y_init,
                    method='RK45',
                    t_eval=tt,
                    # rtol = 1e-10, atol = 1e-12
                    )

    return sol


def simulate_step(tt, init, params):
    """
    Model simulation:
    ------------------------------------
    Returns a Pandas DataFrame of timesteps and populational activity traces with and without optogenetic.
    ------------------------------------
    Parameters:

    tt (running)        - current timepoint;
    init (fixed)        - intial conditions for differential equations;
    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    """

    sol = odesol_step(tt, init, params)

    dd = np.vstack((sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4])).T # Neg
    sim = pd.DataFrame(dd, columns=['t', 'f_e', 'f_pv', 'f_sst', 'f_vip_pos', 'f_vip_neg']) #

    return sim


def residual_step(params, tt, init, data_pc, data_pv, data_sst, data_vip_pos, data_vip_neg,
                  pc_all_err_new, pv_err_new, sst_err_new, vip_pos_err_new, vip_neg_err_new): # 
    """
    Residual function for fitting algorythms:
    ------------------------------------
    Returns flattened and concatenated array of residuals between model simulation and datapoints.
    ------------------------------------
    Parameters:

    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    tt (running)        - current timepoint;
    init (fixed)        - intial conditions for differential equations;
    data_pc             - PC experimental data trace;
    data_pv             - PV experimental data trace;
    data_sst            - SST experimental data trace;
    data_vip            - VIP experimental data trace;

    """
    global count, max_nfev

    #
    # Weights initialization. Weights of 0.05 represents regions of traces smoothed by deconvolution algorythm.
    # Weights of 3 weights_pc[30:38] = 3 were used during initial fittings to represents initial peak in PC population.
    #

    weights_sst = np.float32(np.zeros_like(data_pc))
    weights_vip_pos=np.float32(np.zeros_like(data_pc))
    weights_vip_neg=np.float32(np.zeros_like(data_pc))
    weights_pv = np.float32(np.zeros_like(data_pc))
    weights_pc = np.float32(np.zeros_like(data_pc))

    weights_pv[0:24] = 1.
    weights_pv[24:36] = 0.05
    weights_pv[36:] = 1.  #:93

    weights_sst[0:24] = 1.
    weights_sst[24:30] = 1.0
    weights_sst[30:93] = 3.0
    weights_sst[93:] = 1.0

    weights_vip_pos[2:24]=1.0
    weights_vip_pos[24:30]=1.0
    weights_vip_pos[30:90]=1.0
    weights_vip_pos[90:]=1.0

    weights_vip_neg[2:24]=1.0
    weights_vip_neg[24:30]=1.0
    weights_vip_neg[30:40]=2.0
    weights_vip_neg[30:90]=1.0
    weights_vip_neg[90:]=1.0

    weights_pc[0:24] = 1.
    weights_pc[24:30] = 0.05
    weights_pc[30:38] = 1.
    weights_pc[38:93] = 1.
    weights_pc[93:] = 1.

    #
    # pc_all_err_new, pv_err_new, sst_err_new, vip_err_new are standard errors required by lmfit deffinition of residual functions
    # (check lmfit documentation)
    #

    model = simulate_step(tt, init, params)

    pc_r = (np.float32(np.array(model['f_e'].values - data_pc))
            * weights_pc/pc_all_err_new).ravel()

    pv_r = (np.float32(np.array(model['f_pv'].values - data_pv))
            * weights_pv/pv_err_new).ravel() 

    sst_r = (np.float32(np.array(model['f_sst'].values - data_sst))
             * weights_sst/sst_err_new).ravel() 

    vip_r_pos = (np.float32(np.array(model['f_vip_pos'].values - data_vip_pos))
                 *weights_vip_pos/vip_pos_err_new).ravel() 

    vip_r_neg = (np.float32(np.array(model['f_vip_neg'].values - data_vip_neg))
                 *weights_vip_neg/vip_neg_err_new).ravel() 

    arr = np.concatenate((pc_r, pv_r, sst_r, vip_r_pos, vip_r_neg), axis=0) #

    return arr


def RMSE_full(params, init, data_pc, data_pv, data_sst, data_vip_pos, data_vip_neg, t_exp): #
    """
    Root:Mean Square Error (RMSE) calculation:
    ------------------------------------
    Returns RMSE value for full fit of 4 averages of populations: PC, PV, SST, VIP.
    ------------------------------------
    Parameters:

    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    init (fixed)        - intial conditions for differential equations;
    data_pc             - PC experimental data trace;
    data_pv             - PV experimental data trace;
    data_sst            - SST experimental data trace;
    data_vip            - VIP experimental data trace;

    """
    model = simulate_step(t_exp, init, params)
    sum = 0
    for i in range(len(data_pc)):

        sum += ((model['f_e'].values[i] - data_pc[i]) ** 2
                + (model['f_pv'].values[i] - data_pv[i]) ** 2
                + (model['f_sst'].values[i] - data_sst[i]) ** 2
                + (model['f_vip_pos'].values[i] - data_vip_pos[i])**2 
                + (model['f_vip_neg'].values[i] - data_vip_neg[i])**2)

    sum_norm = np.sqrt(sum / (len(data_pc) * 5)) #neg

    return sum_norm


def RMSE_full_stim(params, init, data_pc, data_pv, data_sst, data_vip_pos, data_vip_neg, t_exp): #
    """
    Root:Mean Square Error (RMSE) calculation:
    ------------------------------------
    Returns RMSE value for full fit of 4 averages of populations: PC, PV, SST, VIP. Calculated only for stimulus interval
    ------------------------------------
    Parameters:

    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    init (fixed)        - intial conditions for differential equations;
    data_pc             - PC experimental data trace;
    data_pv             - PV experimental data trace;
    data_sst            - SST experimental data trace;
    data_vip_pos            - VIP experimental data trace;
    data_vip_neg            - VIP experimental data trace;

    """

    model = simulate_step(t_exp, init, params)
    sum = 0
    for i in range(31, 91):

        sum += ((model['f_e'].values[i] - data_pc[i]) ** 2
                + (model['f_pv'].values[i] - data_pv[i]) ** 2
                + (model['f_sst'].values[i] - data_sst[i]) ** 2
                + (model['f_vip_pos'].values[i] - data_vip_pos[i]) ** 2
                + (model['f_vip_neg'].values[i] - data_vip_neg[i]) ** 2)

    sum_norm = np.sqrt(sum / (len(data_pc) * 5)) #Neg 5

    return sum_norm


def RMSE(params, init, data, type, t_exp):
    """
    Root Mean Square Error (RMSE) calculation:
    ------------------------------------
    Returns RMSE value for one defined fit. Used for optogenetic RMSE calculation
    ------------------------------------
    Parameters:

    params              - lmfit.Parameters() object with all parameters from the differential equations systems
    init (fixed)        - intial conditions for differential equations;
    data                - Typically data for optogenetic effect on PCs;
    type                - Typically "f_e_o" - for optogenetic version of model.

    """

    model = simulate_step(t_exp, init, params)
    sum = 0
    for i in range(31, 91):
        sum += (model[type].values[i] - data[i]) ** 2

    sum = np.sqrt((sum) / len(data))

    return sum

