if __name__ == '__main__':
    import os

    os.chdir('../')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model.model_training import PiValveNeuralNetwork


def get_predictions(model_loc: str, input_layer_dim: int, input_data: pd.DataFrame, limit: int = None):

    model = PiValveNeuralNetwork(input_layer_features=input_layer_dim)
    model.load_state_dict(torch.load(model_loc))
    model.eval()

    test_data = input_data.to_numpy().T
    test_data = torch.tensor(test_data, dtype=torch.float32)
    # make class predictions with the model
    predictions = (model(test_data) > 0.5).int()

    if limit is not None:
        for pos, failure in enumerate(predictions):
            if failure[0].item() == 1 and min(test_data[pos]) > limit:
                predictions[pos][0] = 0

    return predictions

if __name__ == '__main__':

    from data_loaders.high_res_dataitem import DataItemsClient, prepare_engine_data
    import plotly.graph_objects as go
    from constants.authenticator import auth
    from datetime import datetime, timedelta

    data_client = DataItemsClient(auth=auth)

    asset_id = 115964
    date_end = '2023-10-13'
    df_raw, df_pp = prepare_engine_data(asset_id=asset_id, date_end=date_end, data_client=data_client)

    predictions = get_predictions(model_loc='model/pivalve-analytic-5.pt', input_layer_dim=1000, input_data=df_pp, limit = -6)

    for i in range(20):
        # fig = px.scatter(x=list(range(0, len(test_data[random_index]))), y=test_data[random_index])
        # fig.update_layout(title='%d (expected)' % (predictions[random_index]))
        # fig.show()
        print(f'Cylinder %d - prediction %d' % (i + 1, predictions[i]))