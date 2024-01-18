from typing import List

import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, html, dcc
from datetime import datetime, date
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import plotly.express as px
from data_loaders.high_res_dataitem import DataItemsClient, prepare_engine_data
import plotly.graph_objects as go
import yaml
from model.load_model import get_predictions
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

with open('config_analytic.yaml', 'r') as f:
    CONFIG_ANALYTIC = yaml.load(f, Loader=yaml.SafeLoader)

# pi_dataitems = [20310 + i for i in range(20)]
# data_items = pi_dataitems + [161, 102, 107, 20307]
# data_range = 365  # days to download before
# sampling_points = 1000
# interval = 1800
#
# model_location = 'model/pivalve-analytic-5.pt'
# degradation_limit = -6

pi_data_items = [int(x) for x in CONFIG_ANALYTIC['DOWNLOAD']['PI_VALVE_DATAITEMS'].split(',')]
data_items = pi_data_items + [int(x) for x in CONFIG_ANALYTIC['DOWNLOAD']['ADDITIONAL_DATAITEMS'].split(',')]


def create_table(df):
    columns, values = df.columns, df.values
    header = [html.Tr([html.Th(col) for col in columns])]
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in values]
    table = [html.Thead(header), html.Tbody(rows)]
    return table


def convert_cylinders(cylinders):
    return ', '.join(["Cylinder " + str(x + 1) for x in cylinders])


def get_fleet_malfunctions(asset_ids: List, data_client: DataItemsClient):
    warnings = dict()
    alarms = dict()
    date_end = pd.to_datetime('today')
    dis = ([int(x) for x in CONFIG_ANALYTIC['DOWNLOAD']['PI_VALVE_DATAITEMS'].split(',')] +
                  [int(x) for x in CONFIG_ANALYTIC['DOWNLOAD']['ADDITIONAL_DATAITEMS'].split(',')])
    for asset_id in asset_ids:
        _, df_pp = prepare_engine_data(asset_id=asset_id, date_end=date_end, data_client=data_client,
                                       sampling_points=CONFIG_ANALYTIC['DOWNLOAD']['SAMPLING_POINTS'],
                                       data_range=CONFIG_ANALYTIC['DOWNLOAD']['DOWNLOAD_DATA_RANGE'],
                                       data_items=dis,
                                       interval=CONFIG_ANALYTIC['DOWNLOAD']['SAMPLING_INTERVAL'])

        predictions = get_predictions(model_loc=CONFIG_ANALYTIC['MODEL']['MODEL_LOCATION'],
                                      input_layer_dim=CONFIG_ANALYTIC['DOWNLOAD']['SAMPLING_POINTS'],
                                      input_data=df_pp, warning_limit=CONFIG_ANALYTIC['MODEL']['WARNING_LIMIT'],
                                      alarm_limit=CONFIG_ANALYTIC['MODEL']['ALARM_LIMIT'])

        warnings[asset_id] = (predictions == 1).nonzero(as_tuple=True)[0].tolist()
        alarms[asset_id] = (predictions == 2).nonzero(as_tuple=True)[0].tolist()

    df_warnings = pd.DataFrame(list(warnings.items()), columns=['Asset', 'Warnings'])
    df_alarms = pd.DataFrame(list(alarms.items()), columns=['Asset', 'Alarms'])

    df = pd.merge(df_warnings, df_alarms, on='Asset')

    df['Warnings'] = df['Warnings'].apply(convert_cylinders)
    df['Alarms'] = df['Alarms'].apply(convert_cylinders)

    return df


def get_layout():
    return html.Div(children=[
        dcc.Store(id='malfunction-store'),
        html.Div(id='initialise'),
        dmc.Grid(
            children=[
                dmc.Col(
                    dmc.Paper(
                        p=8,
                        m=4,
                        style={"box-shadow": "0 0 1px 1px #bfbfbf"},
                        children=[
                            dmc.Title('Fleet Malfunctions', order=5),
                            dmc.Space(h=10),
                            dcc.Loading(
                                dmc.Table(
                                    striped=True,
                                    highlightOnHover=True,
                                    withBorder=True,
                                    withColumnBorders=True,
                                    id='malfunctions-table',
                                    children=[]
                                )
                            ),
                            dcc.Download(id="download-dataframe-excel"),
                            dmc.Button("Download Data", id='download-malfunctions', mt=25),
                        ]
                    ),
                    span=4
                ),
                dmc.Col(
                    dmc.Paper(
                        p=8,
                        m=4,
                        style={"box-shadow": "0 0 1px 1px #bfbfbf"},
                        children=[
                            dmc.Paper(
                                p=8,
                                m=4,
                                style={"box-shadow": "0 0 1px 1px #bfbfbf"},
                                children=[
                                    dmc.Group(
                                        children=[
                                            dmc.TextInput(label="Asset Id",
                                                          style={"width": 200},
                                                          id='assetid-input',
                                                          value=117084),
                                            dmc.DatePicker(
                                                id="date-picker",
                                                label="End Date",
                                                # description="You can also provide a description",
                                                minDate=date(2020, 8, 5),
                                                value=datetime.now().date(),
                                                style={"width": 200},
                                            ),
                                            dmc.Button("Submit", id='update-button', mt=25),
                                        ]
                                    ),
                                ]
                            ),
                            dmc.Grid(
                                children=[
                                    dmc.Col(
                                        dmc.Paper(
                                            p=8,
                                            m=4,
                                            style={"box-shadow": "0 0 1px 1px #bfbfbf"},
                                            children=[
                                                dmc.Title('PI-Valves Condition', order=5),
                                                dmc.Space(h=10),
                                                dcc.Loading(
                                                    dmc.Stack(
                                                        # spacing="xs",
                                                        children=[
                                                            dmc.Badge(f'Cylinder {cyl}', variant='fill', color='green',
                                                                      id={'name': f'indicator_{cyl}',
                                                                          'type': 'failure-indicators'})
                                                            for cyl in list(range(1, 21))
                                                        ],
                                                    )
                                                )
                                            ]
                                        ),
                                        span=2),
                                    dmc.Col(
                                        children=[
                                            dmc.Paper(
                                                p=8,
                                                m=4,
                                                style={"box-shadow": "0 0 1px 1px #bfbfbf"},
                                                children=[
                                                    dmc.Title('Raw Signals', order=5),
                                                    dmc.Space(h=10),
                                                    dcc.Loading(
                                                        dcc.Graph(id='rawdata-figure',
                                                                  figure=go.Figure().update_layout(
                                                                      margin=dict(l=10, r=10, t=10, b=10),
                                                                      height=300)),
                                                    )
                                                ]
                                            ),
                                            dmc.Paper(
                                                p=8,
                                                m=4,
                                                style={"box-shadow": "0 0 1px 1px #bfbfbf"},
                                                children=[
                                                    dmc.Title('Post-Processed Signals', order=5),
                                                    dmc.Space(h=10),
                                                    dcc.Loading(
                                                        dcc.Graph(id='ppdata-figure',
                                                                  figure=go.Figure().update_layout(
                                                                      margin=dict(l=10, r=10, t=10, b=10),
                                                                      height=300)),
                                                    )
                                                ]
                                            ),
                                            dcc.Store(id='store-data'),
                                            dmc.Button("Download Data", id='download-button', mt=25),
                                            dcc.Download(id="download-csv")
                                        ],
                                        span=10
                                    )
                                ]
                            )
                        ]
                    ),
                    span=8
                )
            ]
        )
        ]
    )


def get_callbacks(app: Dash, data_client: DataItemsClient):
    @app.callback(
        Output('malfunction-store', 'data'),
        Output('malfunctions-table', 'children'),
        Input('initialise', 'children')
    )
    def load_data(_):

        fleet_ids = [115706, 115708, 115807, 115964, 115965, 116249, 116250, 116253, 116259, 116871, 117002, 117057,
                     117081, 117082, 117083, 117084, 117085, 117086, 117087, 117912]
        df = get_fleet_malfunctions(asset_ids=fleet_ids, data_client=data_client)

        return df.to_dict('records'), create_table(df)

    @app.callback(
        Output("download-dataframe-excel", "data"),
        Input("download-malfunctions", "n_clicks"),
        State('malfunction-store', 'data')
    )
    def func(n_clicks, data):
        if n_clicks is None:
            raise PreventUpdate
        else:
            df = pd.DataFrame(data)
            return dcc.send_data_frame(df.to_excel, "mydf.xlsx", sheet_name="Sheet1")
    
    @app.callback(
        Output('rawdata-figure', "figure"),
        Output('ppdata-figure', "figure"),
        Output({'name': ALL, 'type': 'failure-indicators'}, 'color'),
        Output('store-data', 'data'),
        Input('update-button', "n_clicks"),
        Input('date-picker', 'value'),
        State('assetid-input', 'value'),
    )
    def update(_, date_end, asset_id):

        if asset_id is None or date_end is None:
            raise PreventUpdate

        df_raw, df_pp = prepare_engine_data(asset_id=asset_id, date_end=date_end, data_client=data_client,
                                            sampling_points=CONFIG_ANALYTIC['DOWNLOAD']['SAMPLING_POINTS'],
                                            data_range=CONFIG_ANALYTIC['DOWNLOAD']['DOWNLOAD_DATA_RANGE'],
                                            data_items=data_items,
                                            interval=CONFIG_ANALYTIC['DOWNLOAD']['SAMPLING_INTERVAL'])
        df_raw = df_raw.loc[(df_raw[102] > 9300)]

        fig = go.Figure().update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
        fig1 = go.Figure().update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)

        if df_pp.empty:
            return fig, fig1, ['grey' for i in pi_data_items], df_pp.to_dict('records')
        else:

            predictions = get_predictions(model_loc=CONFIG_ANALYTIC['MODEL']['MODEL_LOCATION'],
                                          input_layer_dim=CONFIG_ANALYTIC['DOWNLOAD']['SAMPLING_POINTS'],
                                          input_data=df_pp, warning_limit=CONFIG_ANALYTIC['MODEL']['WARNING_LIMIT'],
                                          alarm_limit=CONFIG_ANALYTIC['MODEL']['ALARM_LIMIT'])

            fig = go.Figure().update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
            fig1 = go.Figure().update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
            fig.add_trace(
                go.Scatter(x=df_raw['time'], y=df_raw[20307], mode='lines', line=dict(color='#1F77B4', width=4),
                           name=f'Cylinder Average'))
            colors = {0: 'grey', 1: 'orange', 2: 'red'}
            for i, di in enumerate(pi_data_items):
                color = colors[predictions[i].item()]
                width = 2 if predictions[i] == 0 else 4
                fig.add_trace(
                    go.Scatter(x=df_raw['time'], y=df_raw[di], mode='lines', line=dict(color=color, width=width),
                               name=f'Cylinder {i + 1}'))
                fig1.add_trace(go.Scatter(x=df_pp.index, y=df_pp[di], mode='lines', line=dict(color=color, width=width),
                                          name=f'Cylinder {i + 1}'))

            return fig, fig1, [colors[prediction.item()] for prediction in predictions], df_pp.to_dict('records')

    @app.callback(
        Output("download-csv", "data"),
        Input("download-button", "n_clicks"),
        State('store-data', 'data'),
    )
    def func(n_clicks, data):
        if n_clicks is None:
            raise PreventUpdate
        df = pd.DataFrame(data)
        columns = dict()
        for cyl, di in enumerate(pi_data_items):
            columns[di] = f'cylinder_{cyl + 1}'
        df = df.rename(columns=columns)
        return dcc.send_data_frame(df.to_csv, "train_data.csv")


if __name__ == '__main__':
    from constants.authenticator import auth
    import plotly.io as pio

    data_client = DataItemsClient(auth=auth)

    pio.templates["alphabet"] = go.layout.Template(
        layout=go.Layout(
            colorway=px.colors.qualitative.Alphabet
        )
    )
    pio.templates.default = 'plotly_white+alphabet'

    app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

    app.layout = get_layout()
    get_callbacks(app=app, data_client=data_client)

    app.run_server(port=9090)
