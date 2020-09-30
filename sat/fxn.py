import numpy as np
import pandas as pd
import plotly.graph_objects as go
import collections
from scipy.signal import find_peaks

Sweep = collections.namedtuple('Sweep', 'dt period rmp hpp hp_amp hpp_s hpp_e')

# functions to analysis sweep
def get_derivative(array, x_unit_distance):
    deriv = (array[1:] - array[:-1])/x_unit_distance
    return deriv


# check large jump on the time series  
def get_index_sharp_change(in_array, threshold=-1, window=1): 
    d = np.diff(in_array)
    index = None
    sense = np.sign(threshold) # 1 if >0, -1 if < 0
    inds = np.where((d-threshold)*sense>=0)[0]
    for ind in inds: 
        difference = in_array[(ind+1):(ind+window+1)] - in_array[ind]
        if np.all((difference-threshold)*sense >= 0):
             index = ind 
             break
    return index 


def local_cumsum(x): # assume x np array of boolean 
    res = np.zeros(len(x))
    if x[0]:
         res[0] =1 
    for i in range(1, len(x)):
        if x[i]:
            res[i] = res[i-1] + 1
    return res


def get_resting_membrane_potential(voltage, threshold=-2, max_id=5000,offset=200, wide=50):
        voltage = np.array(voltage)
        volt_d = np.diff(voltage)
        inds = np.where(volt_d < threshold) [0]
        ind = max_id
        if len(inds) > 0: 
            ind = min(inds[0], ind)
        return ind-offset, ind-offset+wide, voltage[(ind-offset):(ind-offset+wide)].mean()


#HPP
def get_hyperpolarized_potential(voltage,current=None, window=500):
    voltage = np.array(voltage)
    if current is None:
        current = voltage
    hp_s = get_index_sharp_change(current,  threshold= -1, window=1)
    hp_e = get_index_sharp_change(current, threshold=1, window=1)
    hpp = np.amin(voltage[hp_s:hp_e])
    hpp_ind = np.argmax(voltage == hpp)
    return hp_s, hp_e, hpp_ind, hpp
    #hpp = None
    #hp_c = None
    #if hp_e is not None:
        #hp_c = hp_e - window
        #hpp  = voltage[hp_c:hp_e].mean()
    #return hp_s, hp_c, hp_e, hpp


def get_peak_threshold_index(voltage, voltage_derivative, peak_index):
    for index in range(peak_index, 0, -1):
        condition1 = voltage_derivative[index] < voltage_derivative[index+1]
        condition2 = voltage_derivative[index] < voltage_derivative.max() * 0.2 #???
        if condition1 & condition2:
            return index
    return index


def get_repolar_peak_index(voltage, start_i, end_i, lag=100):
    if end_i is None:
        end_i = len(voltage)-1
    lag = min(int((end_i - start_i)/2), lag)  
    repolar_peak_index = np.argmax(voltage[(start_i+lag):end_i]) + start_i+lag
    return repolar_peak_index


def get_repolar_bottom_index(voltage,  start_i):
    repolar_bottom_index = np.argmin(voltage[start_i:]) + start_i
    return repolar_bottom_index

# detect bursts
def get_bursts(peak_indices, time_resolution):
    # burst threshold > 10 Hz, which is equivalent to ptp_interval < 0.1 s (1000 indices)
    ## within_burst_freq = 10
    within_burst_freq = 25
    ptp_threshold = round((1/within_burst_freq) / time_resolution)
    # tonic firing threshold > 2Hz, >= 10 peaks; or 2-10Hz, >= 7 peaks;
    ## within_tonic_freq = 2
    within_tonic_freq = 5
    threshold_tonic = round((1/within_tonic_freq) / time_resolution)
    print(f'ptp_dur : {ptp_threshold}, tonic_dur: {threshold_tonic}')

    # # 1.initial dataframe and calc ptp_interval
    peak_distance = np.diff(peak_indices)
    tmp = pd.DataFrame.from_dict({'peak_index' : peak_indices, 
                'ptp_interval_pre':  np.append(np.nan, peak_distance),
                'ptp_interval_post' : np.append(peak_distance, np.nan)})
    tmp = tmp.assign(is_starting_burst = False,
            is_ending_burst = False,
            is_in_burst = False,
            is_starting_event = False,
            is_less_100ms = False,
            is_less_500ms = False,
            is_starting_tonic = False,
            is_ending_tonic = False,
            is_in_tonic = False, 
            is_less_threshold_tonic = False)
    tmp= tmp.fillna({'ptp_interval_pre': 4e4, 'ptp_interval_post': 4e4})

    # # 2. calculate indicators using ptp_interval 
    mask = (tmp.ptp_interval_pre<ptp_threshold) | (tmp.ptp_interval_post<ptp_threshold)
    tmp.loc[mask, 'is_in_burst'] = True
    mask =  (tmp.ptp_interval_pre >= ptp_threshold) & (tmp.ptp_interval_post < ptp_threshold)
    tmp.loc[mask, 'is_starting_burst']  = True
    #tmp.loc[mask, 'is_starting_event'] = True
    mask = (tmp.ptp_interval_pre < ptp_threshold) & (tmp.ptp_interval_post >= ptp_threshold)
    tmp.loc[mask, 'is_ending_burst']  = True   
    #mask = (tmp.ptp_interval_pre >= ptp_threshold) & (tmp.ptp_interval_post >= ptp_threshold)
    #tmp.loc[mask, 'is_starting_event']  = True

    mask = tmp.ptp_interval_pre < threshold_tonic
    tmp.loc[mask, 'is_less_threshold_tonic'] = True
    tmp['num_previous_less_threshold_tonic'] = local_cumsum(tmp.is_less_threshold_tonic)

    mask = tmp.ptp_interval_pre < ptp_threshold
    tmp.loc[mask, 'is_less_100ms']  = True
    tmp['num_previous_less_100ms'] = local_cumsum(tmp.is_less_100ms) 

    ## 3. detect tonic firing: is_in_tonic true if in 10 rolling window, is_less_threshold_tonic ptp_max/ptp_min < 3  
    tonic_window = 10
    tonic_ptp_ratio = 3 
    for index in range(len(peak_indices)-tonic_window+1):
        condition1 = np.all(tmp['is_less_threshold_tonic'][index:(index+tonic_window)])
        ptp_max = tmp['ptp_interval_pre'][index:(index+tonic_window)].max()
        ptp_min = tmp['ptp_interval_pre'][index:(index+tonic_window)].min()
        condition2 = (ptp_max/ptp_min) < tonic_ptp_ratio
        if condition1 & condition2:
            tmp.loc[(index-1):(index+tonic_window), 'is_in_tonic'] = True

            if tmp['num_previous_less_threshold_tonic'][index-1] == 0:
                tmp.loc[index-1,'is_starting_tonic'] = True

    # Re-adjust is_tonic false if already is_in_burst
    tmp.loc[tmp.is_in_burst, 'is_in_tonic'] =False
    mask = (tmp.is_starting_burst) & ( ~tmp.is_in_burst)
    tmp.loc[mask, 'is_starting_burst' ] = False
    ## 4. round 2 clear up burst 
    ### remove burst if ibi > 5 sec ??? everything after ? 
    #mask = tmp.is_starting_burst & tmp.ptp_interval_pre > 5/time_resolution
    #if any(mask):
    #    tmp.loc[mask, 'is_starting_burst'] = False
    #    tmp.loc[mask:, 'is_in_burst'] = False
    ### Remove short ones
    ### Cut long ones
    ### check if missed one peak in the bursts
    ### Add peak frequency
    tmp['frequency'] = round(1/(tmp['ptp_interval_post'] *time_resolution))

    ### Add burst_index
    tmp['burst_index']= tmp.is_starting_burst.cumsum()
    tmp.loc[~tmp.is_in_burst, 'burst_index'] = -1 # not in burst, should have negative burst index 
    df_peak_info= tmp
    print(f'detected {max(df_peak_info.burst_index)} bursts')
    return df_peak_info


def set_busrt_index(df_peak_info):
    np.where(df_peak_info.is_starting_burst)

def get_burst_detail(current_burst, time, voltage, voltage_derivative, sweep_info, end_index=None): 
    # get burst detail detail 
    sample_freq = 1/sweep_info.dt
    num_ap_in_burst = len(current_burst)
    i_first_peak = current_burst.peak_index.min()
    i_last_peak = current_burst.peak_index.max()
    burst_duration = (time[i_last_peak] - time[i_first_peak])*sample_freq
    ap_frequency_in_burst = (num_ap_in_burst-1)/burst_duration
    i_ap_threshold = get_peak_threshold_index(voltage, voltage_derivative, i_first_peak)
    ap_threshold = voltage[i_ap_threshold]

    latency = (time[i_ap_threshold] - time[sweep_info.hpp_e]) # *sample_freq

    ahp_bottom = np.argmin(voltage[i_last_peak:end_index]) + i_last_peak
    ahp_end = np.argmax(voltage[ahp_bottom:end_index] >  sweep_info.rmp) + ahp_bottom

    burst_adp_peak = get_repolar_peak_index(voltage, i_last_peak, ahp_bottom)      
    burst_adp_bottom = get_repolar_bottom_index(voltage, burst_adp_peak)
    ahp = voltage[burst_adp_peak] - voltage[burst_adp_bottom]
    ahp_start = np.argmax(voltage[burst_adp_peak:] < sweep_info.rmp) + burst_adp_peak
    ahp_bottom_voltage = voltage[np.argmin(voltage[ahp_start:ahp_end]) + ahp_start]
    ahp_amplitude = sweep_info.rmp - voltage[ahp_bottom]
    ahp_time2 = (ahp_end - ahp_start)*sweep_info.dt

    ahp_slope = (voltage[ahp_start] - voltage[burst_adp_peak]) / (time[ahp_start] - time[burst_adp_peak]) /1000
    print(f'Burst: ap_threshold={i_ap_threshold}, first peak={i_first_peak}, last_peak={ i_last_peak}, hpp_e={sweep_info.hpp_e}, repolar_peak={burst_adp_peak}')
    print(f'Burst: duration={burst_duration:.2f}, frequency={ap_frequency_in_burst:.2f}, latency={latency:.2f}, ahp={ahp:.2f}, ahp_time={ahp_time2:.2f}')
    print(f'ahp start/end: {ahp_start}, {ahp_end}, {ahp_bottom}, slope: {ahp_slope:.2f} mV/s, amplitude: {ahp_amplitude:.2f} mV, min voltage: {ahp_bottom_voltage:.2f} mV, duration: {ahp_time2:.2f}s')

    intraburst_ahp = []
    for i in range(len(current_burst.peak_index)-1):
         intraburst_ahp.append(np.argmin(voltage[current_burst.peak_index.iloc[i]:current_burst.peak_index.iloc[i+1]]) +  current_burst.peak_index.iloc[i])

    intraburst_ahp.append(np.argmin(voltage[i_last_peak:burst_adp_peak]) + i_last_peak)
    intraburst_ahp = np.array(intraburst_ahp)
    
    intraburst_ahp_voltage = voltage[intraburst_ahp]
    intraburst_ahp_amplitude = voltage[current_burst.peak_index] - voltage[intraburst_ahp]
    print(f'intraburst_ahp: {intraburst_ahp},\nintraburst_ahp_voltage: {intraburst_ahp_voltage}\nintraburst_ahp_amplitude: {intraburst_ahp_amplitude}')
    #\nintraburst_ahp_amplitude: {intraburst_ahp_amplitude}

    plot_points = [i_ap_threshold, burst_adp_peak, ahp_bottom, ahp_start, ahp_end]
    point_labels = ['ap_threshold', 'adp_peak', 'ahp_bottom', "ahp_start", "ahp_end"]
    plot_df = pd.DataFrame({'point_label': point_labels, 'point_index': plot_points})
    return {'duration': burst_duration, 'ap_freq': ap_frequency_in_burst, 'latency': latency, 'ahp': ahp, 
                'ahp_start': ahp_start, 'ahp_end': ahp_end, 'ahp_slope': ahp_slope,  'ahp_amp': ahp_amplitude, 
                'ahp_min_vol': ahp_bottom_voltage, 'ahp_duration': ahp_time2, 'intra_ahp': intraburst_ahp, 
                'intra_ahp_vol': intraburst_ahp_voltage, 'plot_df': plot_df,
                'burst_points': [i_ap_threshold, i_first_peak, i_last_peak, sweep_info.hpp_e, burst_adp_peak] 
     }


# graph function for sweep 
def plot_sweep(time, current, voltage):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=current,
                        mode='lines',
                        name='current'))
    fig.add_trace(go.Scatter(x=time, y=voltage,
                        mode='lines',
                        name='voltage'))
    return fig

def plot_hpp(time, current, voltage, hpp_df):
    fig = go.Figure()
    plot_points = hpp_df.point_index
    point_labels = hpp_df.label

    start = min(plot_points)-1000
    end = max(plot_points)+1000
    fig.add_trace(go.Scatter(x=time[start:end], y=voltage[start:end],
                        mode='lines',
                        name='voltage'))
    fig.add_trace(go.Scatter(x=time[start:end], y=current[start:end],
                        mode='lines',
                        name='current'))                    
    fig.add_trace(go.Scatter(x=time[plot_points], y=voltage[plot_points],
                        mode='markers', text=point_labels,
                        name='hpp points'))
    return fig

def plot_burst(time, voltage, sweep_info, current_burst, burst_detail):
    fig = go.Figure()
    start = max(min(current_burst.peak_index)-100, 0)
    end = min(burst_detail['ahp_end']+100, len(voltage)-1)
    plot_points = burst_detail['plot_df'].point_index
    point_labels =  burst_detail['plot_df'].point_label
    rmp = sweep_info.rmp
    intraburst_ahp = burst_detail['intra_ahp']

    fig.add_trace(go.Scatter(x=time[start:end], y=voltage[start:end],
                        mode='lines',
                        name='voltage'))
    fig.add_trace(go.Scatter(x=time[current_burst.peak_index], y=voltage[current_burst.peak_index],
                        mode='markers',
                        name='peaks'))
    fig.add_trace(go.Scatter(x=time[intraburst_ahp], y=voltage[intraburst_ahp],
                        mode='markers',
                        name='intra_ahp_bottom'))                     
    fig.add_trace(go.Scatter(x=time[plot_points], y=voltage[plot_points],
                        mode='markers', text=point_labels,
                        name='ahp points'))     
    fig.add_shape(# Line Horizontal
                type="line", x0=time[start], y0=rmp, x1=time[end], y1=rmp,
                line=dict(
                    color="LightSeaGreen",
                    width=1,
                    dash="dashdot",
                ),
        ) 
    return fig

# method to handle a sweep of abf file 
def process_sweep(abf, sweep, name, save_plt=False):
    abf.setSweep(sweep)
    time = abf.sweepX
    time_resolution =  time[1]  
    freq = round(1/time_resolution) 
    voltage = abf.sweepY
    current = abf.sweepC
    voltage_derivative = get_derivative(voltage, x_unit_distance = time_resolution)
    print(f'\n***********trace: {name}, sweep: {sweep}***********')
    print(f'freq : {freq} Hz, sample time: {time_resolution*1000} ms, period: {round(time[-1],2)} sec')

    # RMP 
    rmp_s, rmp_e, rmp = get_resting_membrane_potential(voltage)
    print(f'RMP: {rmp:.2f} mV, averaging from {rmp_s} - {rmp_e}')

    # stimulus start end 
    stim_s = get_index_sharp_change(current, threshold=-10, window=1)
    stim_e = get_index_sharp_change(current, threshold=10, window=1)
    print(f'Stimulus starts at {stim_s} end at {stim_e}')

    # hyperpolarization
    hpp_s, hpp_c, hpp_e, hpp = get_hyperpolarized_potential(voltage, current)
    hp_amp = hpp - rmp
    print(f'RMP={rmp:.2f} mV, HPP={hpp:.2f} mV, HP_AMP={hp_amp:.2f} mV, Current Amplitude={current.min()} mA')
    print(f'Hyperpolarization start at {hpp_s}, end at {hpp_e}')

    hpp_points = [rmp_s, rmp_e, hpp_s, hpp_c, hpp_e]
    hpp_labels = ['rmp_start', 'rmp_end', 'hpp_start', 'hpp_calc', 'hpp_end']
    hpp_df = pd.DataFrame({'label': hpp_labels, 'point_index': hpp_points})

    sweep_info = Sweep(dt=time_resolution, period=round(time[-1],2), rmp=rmp, 
                hpp=hpp, hp_amp=hp_amp, hpp_s=hpp_s, hpp_e=hpp_e)

    # find peaks 
    peak_indices, _ = find_peaks(voltage, height=-20, threshold=-40)
    peak_indices = np.array(peak_indices)
    if len(peak_indices) > 0: 
        print(f'peak num: {len(peak_indices)},  peak start: {peak_indices[0]}, peak end: {peak_indices[-1]}, hyperpolarization end: {hpp_e}')
        print(f'peak starts after hpp ends: {hpp_e < peak_indices[0]}')
        print(f'peak voltages: {voltage[peak_indices]}')
    else:
        print('No peak detected')
        return sweep_info, None, None

    # get bursts info 
    df_peak_info = get_bursts(peak_indices, time_resolution)

    figs = []
    if save_plt:
        full_plot = plot_sweep(time, current, voltage)
        full_plot.update_layout(
            title=name+'_'+str(sweep),
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        hpp_plot = plot_hpp(time, current, voltage, hpp_df)
        hpp_plot.update_layout(
            title=name+'_'+str(sweep) + '_hpp',
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )

        figs = [full_plot, hpp_plot]

    burst_infos = {}
    # get burst detail 
    df_burst_info = df_peak_info.loc[df_peak_info.burst_index > 0]
    num_bursts = max(df_burst_info.burst_index)
    for i in range(num_bursts):
        current_burst_index = i +1 
        current_burst  = df_burst_info[df_burst_info['burst_index'] == current_burst_index]
        if current_burst_index < num_bursts:
            next_burst_start = df_burst_info[df_burst_info['burst_index'] == (current_burst_index+1)].peak_index.iloc[0]
        else:
            next_burst_start = len(voltage)-1
        print(f'saving burst {current_burst_index}, ends {next_burst_start}')
        burst_detail = get_burst_detail(current_burst, time, voltage, voltage_derivative, sweep_info, next_burst_start) 
        burst_infos[current_burst_index]= burst_detail
        if save_plt:
            burst_fig =  plot_burst(time, voltage, sweep_info, current_burst, burst_detail)
            btitle = name+'_'+str(sweep)+"_burst_"+ str(current_burst_index)
            burst_fig.update_layout(
                title=btitle,
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="RebeccaPurple"
                )
            )
            figs.append(burst_fig)
        #burst_fig.write_image(output_data_dir + btitle +".pdf")

    return sweep_info, df_peak_info, burst_infos, figs

def save_plots(plt_dir, name, sweep, figs):
        # save figure to html file 
        file_name = plt_dir + '/trace_' + name +'_sweep_' + str(sweep)
        html_file =  file_name + '.html'
        with open(html_file, 'a') as f:
            for fig in figs:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
