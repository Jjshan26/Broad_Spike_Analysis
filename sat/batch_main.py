import sys
import os
import pandas as pd
import numpy as np
import pyabf
import plotly.express as px
import plotly.graph_objects as go
# add the package root to your path if not already 
sys.path.append('c:/broad/spike-analysis-tool/')
import sat.fxn as mabf
import pickle

USAGE = f"Usage: python {sys.argv[0]} input_dir output_dir [save_plot=True]"

def process_dir(input_data_dir, output_data_dir, save_plot=True):
    print(f'Input from: {input_data_dir}, Output to: {output_data_dir}, Plots: {save_plot} ')
    filenames = os.listdir(input_data_dir)
    for filename in filenames:
        filepath = input_data_dir + '/' + filename

        abf = pyabf.ABF(filepath)
        base = os.path.basename(filepath)
        name = os.path.splitext(base)[0]
        print(f'processing {name}')
        for sweep in range(abf.sweepCount):
            sweep_info, df_peak_info, burst_infos, figs = mabf.process_sweep(abf, sweep, name, save_plot)
            if save_plot:
                mabf.save_plots(output_data_dir, name, sweep, figs)
            
            with open(output_data_dir + name+'sweep_' +str(sweep) + '.pkl', 'wb') as ofp:
                pickle.dump({'sweep' : sweep_info, 'peaks' : df_peak_info, 'bursts': burst_infos}, ofp)
                
            # to collect burst info dataframe
            if burst_infos is not None:
                sweep_bursts = {}
                for index, info in burst_infos.items():
                    info.pop('plot_df')
                    info.pop('burst_points')
                    sweep_bursts[index]= pd.DataFrame(info)

                b_res = pd.concat(sweep_bursts)
                b_res.to_csv(output_data_dir + name+'sweep_' +str(sweep) + '.csv')

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(USAGE)

    input_data_dir = args[0]
    output_data_dir = args[1]
    save_plot= False
    if len(args) > 2:
        save_plot = args[2]=='True' 

    #input_data_dir = "C:/work/broad/input/18/"
    #output_data_dir = "C:/work/broad/output/18/"
    process_dir(input_data_dir, output_data_dir, save_plot)