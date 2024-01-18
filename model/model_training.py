import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class PiValveNeuralNetwork(nn.Module):
    def __init__(self, input_layer_features):
        self.input_layer_features = input_layer_features
        super().__init__()
        # Define Layers:
        self.l1 = nn.Linear(self.input_layer_features, 256)  # layer 1
        self.l2 = nn.Linear(256, 256)  # layer 2
        self.l3 = nn.Linear(256, 64)  # layer 3
        self.l4 = nn.Linear(64, 1)  # layer 4
        # Define Activation functions:
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Layers: 4
        Activation Functions:
        RELU for first two layers
        Sigmoid for third layer
        Log Softmax for last layer
        """
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':

    dataset = np.loadtxt('new_dataset.csv', delimiter=',').T
    input_layer_features = len(np.loadtxt('new_dataset.csv', delimiter=',')) - 1

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    np.random.shuffle(dataset)
    train_dataset, test_dataset = dataset[:train_size, :], dataset[train_size:,:]

    # Convert, create a tensor out of NumPy arrays
    X_train = torch.tensor(train_dataset[:, : input_layer_features], dtype=torch.float32)
    y_train = torch.tensor(train_dataset[:, input_layer_features:], dtype=torch.float32).reshape(-1, 1)

    X_test = torch.tensor(test_dataset[:, : input_layer_features], dtype=torch.float32)
    y_test = torch.tensor(test_dataset[:, input_layer_features:], dtype=torch.float32).reshape(-1, 1)

    model = PiValveNeuralNetwork(input_layer_features=input_layer_features)

    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    n_epochs = 150
    batch_size = 10

    for epoch in range(n_epochs):
        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train[i:i + batch_size]
            y_pred = model(Xbatch)
            ybatch = y_train[i:i + batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    with torch.no_grad():
        y_pred = model(X_train)
    accuracy = (y_pred.round() == y_train).float().mean()
    print(f"Training Accuracy {accuracy}")

    with torch.no_grad():
        y_pred = model(X_test)
    accuracy = (y_pred.round() == y_test).float().mean()
    print(f"Training Accuracy {accuracy}")

    torch.save(model.state_dict(), 'pivalve-analytic-7.pt')

    test_data = np.loadtxt('test_data.csv', delimiter=',').T
    test_data = torch.tensor(test_data, dtype=torch.float32)
    # make class predictions with the model
    predictions = (model(test_data) > 0.5).int()
    for i in range(20):
        # fig = px.scatter(x=list(range(0, len(test_data[random_index]))), y=test_data[random_index])
        # fig.update_layout(title='%d (expected)' % (predictions[random_index]))
        # fig.show()
        print(f'Cylinder %d - prediction %d' % (i + 1, predictions[i]))

