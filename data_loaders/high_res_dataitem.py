if __name__ == '__main__':
    import os

    os.chdir('../')

from pydantic import BaseModel, validator, ValidationError
import requests
from constants.authenticator import Authenticator
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import List, Optional
from rms_commons.utils.decorators import safe_request_decorator, safe_run_function_decorator
import dask.dataframe as dd


class DataItemQuery(BaseModel):
    asset_id: int
    data_items_ids: str
    from_ts: int
    to_ts: Optional[int] = None
    interval: Optional[int] = 1
    last_connected: Optional[int] = None

    @validator('from_ts')
    def validate_from_ts(cls, v):
        if not v:
            raise ValidationError('from_ts is required')
        elif not isinstance(v, int):
            try:
                v = int(v)
            except ValueError:
                raise ValidationError('from_ts needs to be an integers')
        # if len(str(v)) != len_timestamp():
        #     raise ValidationError('to_ts needs to be in ms')
        return v


def current_milli_time():
    return round(time.time() * 1000)


@dataclass
class DataItemsClient:
    auth: Authenticator

    @safe_request_decorator(n_tries=20)
    def _request(self, parameters: dict, asset_id: int) -> dict:
        r = requests.get(url=f"https://myplant.io/app/api/asset/{str(asset_id)}/history/batchdata",
                         params=parameters,
                         headers={'x-seshat-token': self.auth.token})
        r.raise_for_status()
        return r.json()

    def get_dataitems(self, query: DataItemQuery) -> pd.DataFrame:
        parameters = {
            'assetType': 'J-Engine',
            'dataItemIds': query.data_items_ids,
            'from': query.from_ts,
            'timeCycle': query.interval,
            'to': query.to_ts,
        }
        pd.DataFrame()
        results = self._request(parameters=parameters, asset_id=query.asset_id)
        try:
            data_temp = np.hstack([np.reshape(np.vstack(results['data'])[:, 0], (-1, 1)),
                                   np.reshape(np.array(np.vstack(results['data'])[:, 1].tolist(), dtype=object),
                                              (-1, len(results['columns'][1])))])
            data = pd.DataFrame(data_temp, columns=['time'] + results['columns'][1])
            data['time'] = pd.to_datetime(data['time'], unit='ms')
            data['asset_id'] = query.asset_id
            return data
        except BaseException:
            print(f'Failed Request')


@dataclass
class DataItemClient:
    queries: List[DataItemQuery] = field(default_factory=list)
    data_items_client: Optional[DataItemsClient] = field(default_factory=lambda: DataItemsClient())

    @safe_run_function_decorator(n_tries=10)
    def download(self) -> dd.DataFrame:
        df = dd.from_map(self.data_items_client.get_dataitems, self.queries, enforce_metadata=False)
        return df.compute()


def create_query_list(asset_id, data_items_ids, start_time, end_time, interval: Optional[int] = 1):
    queries = []
    from_ts = start_time
    while from_ts < end_time:
        to_ts = from_ts + 3600000*interval
        if to_ts > end_time:
            to_ts = end_time
        queries.append(
            DataItemQuery(
                asset_id=asset_id,
                data_items_ids=','.join(map(str, data_items_ids)),
                from_ts=from_ts,
                to_ts=to_ts,
                interval=interval
            )
        )
        from_ts=from_ts+3600000*interval+1000
    return queries


def to_timestamp(dt: pd.Timestamp, coerce: bool = True):
    """Converts pandas datetime object to timestamp in ms"""
    if isinstance(dt, pd.Timestamp):
        return int(round(dt.value / 1e6))
    elif coerce is False:
        raise ValueError(f'Input to to_timestamp must be a pd.Timestamp object. Received instead {type(dt)}')


def prepare_engine_data(asset_id: int, date_end: str, data_client: DataItemsClient, interval: int,
                        sampling_points: int, data_range: int, data_items: List):

    date_end = pd.to_datetime(date_end)

    queries = create_query_list(asset_id, data_items, to_timestamp(date_end) - data_range * 24 * 3600 * 1000,
                                to_timestamp(date_end), interval=interval)

    df_raw_ = DataItemClient(queries=queries, data_items_client=data_client).download()

    df_pp_ = df_raw_.set_index('time').apply(pd.to_numeric)
    df_pp_ = df_pp_.loc[(df_pp_[102] > 9300) & (df_pp_[data_items].ne(0).any(1))]
    df_pp_ = df_pp_[df_pp_ > 10].dropna()
    df_pp_ = df_pp_.drop(columns=[161, 102, 107, 'asset_id'])

    for column in df_pp_.columns:
        df_pp_[column] = df_pp_[column].ewm(alpha=0.01).mean()

    df_pp_ = df_pp_.sub(df_pp_[20307], axis=0).drop(columns=[20307]).reset_index(drop=True)

    df_pp_ = df_pp_.tail(sampling_points).reset_index(drop=True)
    if len(df_pp_) < sampling_points:
        df_pp_ = pd.DataFrame()
    else:
        df_pp_ = df_pp_ - df_pp_.iloc[0]

    return df_raw_, df_pp_.reset_index(drop=True)


if __name__ == "__main__":
    import yaml
    with open('config_analytic.yaml', 'r') as f:
        CONFIG_ANALYTIC = yaml.load(f, Loader=yaml.SafeLoader)
    from model.load_model import get_predictions

    import plotly.graph_objects as go
    from constants.authenticator import auth
    from datetime import datetime, timedelta

    data_client = DataItemsClient(auth=auth)

    # asset_id = 117084



    df = get_fleet_malfunctions(asset_ids=asset_ids)

    # Convert dictionaries to dataframes


    # Merge dataframes on 'Asset'


    # queries = create_query_list(asset_id, [58, 107], to_timestamp(date_start),
    #                             to_timestamp(date_end), interval=30)
    #
    # df_raw_ = DataItemClient(queries=queries, data_items_client=data_client).download().apply(pd.to_numeric).drop(columns='asset_id').reset_index(drop=True)
    # # df_raw_ = df_raw_.loc[df_raw_[161] > 3000]
    #
    # # Assuming df is your DataFrame and 'col' is your column
    # df_raw_['counter'] = 0
    # counter = 0
    # speed_limit = 1450  # replace with your certain value
    #
    # for i in range(1, len(df_raw_)):
    #     if df_raw_.loc[i, 107] >= speed_limit and df_raw_.loc[i - 1, 107] < speed_limit:
    #         counter += 1
    #     df_raw_.loc[i, 'counter'] = counter
    #
    # df_grouped = df_raw_.groupby('counter').max()
    #
    # fig = go.Figure()
    # for i, column in enumerate([58]):
    #
    #     fig.add_trace(go.Scattergl(x=df_grouped.index, y=df_grouped[column], mode='markers', name=column))
    # # fig.add_trace(go.Scatter(x=df1['time'], y=df1['delta_ewm'], mode='lines', name='valve_duration'))
    # fig.show()


    # def get_data(asset_id: int, date_end: str):
    #     df_raw, df_pp = prepare_engine_data(asset_id=asset_id, date_end=date_end, data_client=data_client)
    #
    #     df_pp = df_raw.set_index('time').apply(pd.to_numeric)
    #     df_pp = df_pp.loc[(df_pp[102] > 9300) & (df_pp[data_items].ne(0).any(1))]
    #     df_pp = df_pp[df_pp > 10].dropna()
    #     df_pp = df_pp.drop(columns=[161, 102, 107, 'asset_id'])
    #
    #     for column in df_pp.columns:
    #         df_pp[column] = df_pp[column].ewm(alpha=0.01).mean()
    #
    #     df_pp = df_pp.sub(df_pp[20307], axis=0).drop(columns=[20307]).reset_index(drop=True)
    #
    #     df_pp = df_pp.tail(sampling_points).reset_index(drop=True)
    #     if len(df_pp) < sampling_points:
    #           df_pp = pd.DataFrame()
    #     else:
    #         df_pp = df_pp - df_pp.iloc[0]
    #     return df_pp, df_raw
    # pi_dataitems = [20310 + i for i in range(20)]
    # data_items = pi_dataitems + [161, 102, 107, 20307]
    # df_raw, df_pp = prepare_engine_data(asset_id=asset_id, date_end=date_end, data_client=data_client,
    #                                     interval=1800, data_items=data_items, sampling_points=1000, data_range=365)

    # fig = go.Figure().update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    # fig1 = go.Figure().update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    # for i, di in enumerate([20310 + i for i in range(20)]):
    #     fig.add_trace(go.Scattergl(x=df_raw.time, y=df_raw[di], mode='lines',
    #                              name=f'Cylinder {i + 1}'))
    #     fig1.add_trace(go.Scattergl(x=df_pp.index, y=df_pp[di], mode='markers',
    #                              name=f'Cylinder {i + 1}'))
    #
    # fig.show()
    # fig1.show()

    # training_data = pd.DataFrame()
    # end_date = datetime.strptime(date_end, '%Y-%m-%d')
    # start_date = end_date - timedelta(days=40)
    # time_delta = timedelta(days=10)
    # current_date = start_date
    # while current_date < end_date:
    #     date_end = current_date.strftime('%Y-%m-%d')
    #     df_raw, df_pp = prepare_engine_data(asset_id=asset_id, date_end=date_end, data_client=data_client)
    #     training_data = pd.concat([training_data, df_pp.add_suffix('_' + date_end)], axis=1)
    #     current_date += time_delta
    #
    # # training_data = training_data.T.reset_index(drop=True).T
    # fig = go.Figure()
    # for i, column in enumerate(training_data.columns):
    #
    #     fig.add_trace(go.Scattergl(x=training_data.index, y=training_data[column], mode='lines', name=column))
    # # fig.add_trace(go.Scatter(x=df1['time'], y=df1['delta_ewm'], mode='lines', name='valve_duration'))
    # fig.show()
    # training_data.to_csv(f'new_training_data_{asset_id}.csv')