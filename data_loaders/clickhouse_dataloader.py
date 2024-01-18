from typing import List

if __name__ == '__main__':
    import os

    os.chdir('../')

import pandas as pd
from enum import Enum
from data_loaders.clickhouse import ClickhouseQueryExecutor

from config import config


def to_timestamp(dt: pd.Timestamp, coerce: bool = True):
    """Converts pandas datetime object to timestamp in ms"""
    if isinstance(dt, pd.Timestamp):
        return int(round(dt.value / 1e6))
    elif coerce is False:
        raise ValueError(f'Input to to_timestamp must be a pd.Timestamp object. Received instead {type(dt)}')

class Aggregations(Enum):
    MIN = 'Minimum'
    MAX = 'Maximum'
    AVERAGE = 'Average'
    PERCENTILE = 'Percentile'


class DataLoader:

    def __init__(self):

        self.ch_executor_myplant = ClickhouseQueryExecutor('myplant')
        self.high_res_time_series_table_name = config['CLICKHOUSE']['DATABASES']['MYPLANT']['TABLE_NAMES']['HIGH_RES_TIME_SERIES']
        self.low_res_time_series_table_name = config['CLICKHOUSE']['DATABASES']['MYPLANT']['TABLE_NAMES']['LOW_RES_TIME_SERIES']


    def get_data_items_data(self, asset_id, ts_from: int, ts_to: int, data_item_ids: list[str]):

        data_item_query = f""" SELECT
                                data_value,
                                data_item_id,
                                data_time,
                                device_id
                            FROM myplant.{self.high_res_time_series_table_name}
                            WHERE
                                device_id = {asset_id} AND
                                data_item_id IN {tuple(data_item_ids)} AND
                                data_time <= {ts_to} AND
                                data_time > {ts_from}"""

        df = self.ch_executor_myplant.execute(data_item_query)

        return df

    # def prepare_engine_data(self, asset_id: int, date_end: str, data_client: DataLoader, interval: int,
    #                             sampling_points: int, data_range: int, data_items: List):
    #
    #     date_end = pd.to_datetime(date_end)
    #     date_start = pd.to_datetime(date_end) - pd.to_timedelta(data_range, 'days')
    #
    #     df = self.get_data_items_data(asset_id=asset_id, ts_from=to_timestamp(date_start), ts_to=to_timestamp(date_end),
    #                                   data_item_ids=data_items).sort_values('data_time')
    #
    #     df_raw_ = df.pivot(index='data_time', columns='data_item_id', values='data_value').interpolate().resample(
    #         f'{interval}s').mean().dropna()
    #
    #     df_pp_ = df_raw_.loc[df_raw_[102] > 9300].dropna()
    #     #
    #
    #     df_pp_ = df_pp_.apply(lambda x: x.ewm(alpha=0.01).mean())
    #     if len(df_pp_) > sampling_points:
    #         df_pp_ = df_pp_.tail(sampling_points).reset_index(drop=True)
    #         df_pp_ = df_pp_.sub(df_pp_[20307], axis=0)
    #         df_pp_ = df_pp_ - df_pp_.iloc[0]
    #
    #     return df_raw_, df_pp_


if __name__ == "__main__":
    import plotly.graph_objects as go

    data_client = DataLoader()

    asset_id = 115964
    end_time = '2024-01-05'
    # start_time = to_timestamp(pd.to_datetime('2023-08-20'))  / 1000
    pi_dataitems = [20310 + i for i in range(20)]
    data_items = [161, 102, 107, 20307]

    date_end = to_timestamp(pd.to_datetime(end_time)) / 1000
    date_start = to_timestamp(pd.to_datetime(end_time) - pd.Timedelta(30, 'days')) / 1000

    df = data_client.get_data_items_data(asset_id=asset_id, ts_from=date_start,
                                         ts_to=date_end,
                                         data_item_ids=data_items+pi_dataitems).sort_values('data_time')

    df_raw_ = df.pivot(index='data_time', columns='data_item_id', values='data_value')
    df_pp_ = df_raw_.loc[df_raw_[102] > 9300].interpolate() #.resample(f'{1800}s').mean().interpolate()
    df_pp_ = df_pp_.drop(columns=[102,107,161])
    df_pp_ = df_pp_.apply(lambda x: x.ewm(alpha=0.01).mean())
    if len(df_pp_) > 1000:
        df_pp_ = df_pp_.tail(1000) #.reset_index(drop=True)
        df_pp_ = df_pp_.sub(df_pp_[20307], axis=0)
        df_pp_ = df_pp_.drop(columns=[20307])
        df_pp_ = df_pp_ - df_pp_.iloc[0]
    # df = data_client.get_data_items_data(asset_id=asset_id, ts_from=to_timestamp(date_start),
    #                                      ts_to=to_timestamp(date_end),
    #                                      data_item_ids=data_items).sort_values('data_time')
    # df = data_client.get_data_items_data(asset_id=asset_id, ts_from=start_time, ts_to=end_time,
    #                                                 data_item_ids=data_items).sort_values('data_time')
    #
    # df_pivot = df.pivot(index='data_time', columns='data_item_id', values='data_value').interpolate()

    # df_pivot = df_pivot.loc[df_pivot[102] > 9300].dropna()
    # #
    #
    # df_pivot = df_pivot.apply(lambda x: x.ewm(alpha=0.01).mean())
    # df_pivot = df_pivot.tail(1000).reset_index(drop=True)
    # df_pivot = df_pivot.sub(df_pivot[20307], axis=0)
    # df_pivot = df_pivot - df_pivot.iloc[0]

    # df_raw, df_pp = prepare_engine_data(asset_id=asset_id, date_end=end_time, data_client=data_client,
    #                                     interval=1800, sampling_points=1000, data_range=365, data_items=pi_dataitems+data_items)

    fig = go.Figure().update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
    for i, di in enumerate(pi_dataitems):
        fig.add_trace(go.Scattergl(x=df_raw_.index, y=df_raw_[di], mode='lines',
                                 name=f'Cylinder {i + 1}'))

    fig.show()