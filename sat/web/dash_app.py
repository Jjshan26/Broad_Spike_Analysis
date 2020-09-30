import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pyabf
import os, sys
# add the package root to your path if not already 
sys.path.append('c:/broad/spike-analysis-tool/')
import sat.fxn as mabf

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def get_files(input_dir):
    filenames = []
    if os.path.exists(input_dir):
        filenames = os.listdir(input_dir)
    return filenames

def process_file(input_dir, filename):
    filepath = input_dir + '/' + filename
    print(f'processing {filepath}')
    abf = pyabf.ABF(filepath)
    base = os.path.basename(filepath)
    name = os.path.splitext(base)[0]
    sweep = 0
    sweep_info, df_peak_info, burst_infos, figs = mabf.process_sweep(abf, sweep, name, True)
    return  sweep_info, df_peak_info, burst_infos, figs

app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('DASH - SPIKE ANALYSIS'),
                                 html.P('Pick a directory to analysis.'),
                                 dcc.Input(id='input_dir', value='C:/work/Broad/input/18/', type='text'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='files_dropdown', 
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='stockselector'
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'})
                                ]
                             ),
                    html.Div(id='charts', className='eight columns div-for-charts bg-grey')
                ]
            )
        ]
)

@app.callback(
    Output(component_id='files_dropdown', component_property= 'options'),
    [Input(component_id='input_dir', component_property='value')]
)
def update_file_dropdown(input_value):
    global in_dir 
    in_dir = input_value
    return [{'label': i, 'value': i} for i in get_files(input_value)]


@app.callback(
    Output(component_id='charts', component_property= 'children'),
    [Input(component_id='files_dropdown', component_property='value')]
)
def update_graphs(file_name):
    sweep_info, df_peak_info, burst_infos, figs = process_file(in_dir, file_name)
    outs = [ html.H2('Spike Analysis') ] + [dcc.Graph(figure=fig) for fig in figs] 
    return outs

if __name__ == '__main__':
    app.run_server(debug=True)