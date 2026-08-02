import torch
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn

from collections import deque
import random

from config import *

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)

        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

        self.act = nn.SiLU()

    def forward(self, x):

        residual = x

        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.ln2(x)

        x = x + residual

        return self.act(x)

class ResidualMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(3, 64),
            nn.SiLU(),

            ResidualBlock(64),
            ResidualBlock(64),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_residual_mlp(
    X,
    Y,
    train_seconds,
    dt=0.01,
    epochs=200,
    lr=1e-3,
):

    n_train = int(train_seconds / dt)

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)

    model = ResidualMLP()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.MSELoss()

    for epoch in range(epochs):

        optimizer.zero_grad()

        prediction = model(X_train)

        loss = criterion(prediction, Y_train)

        loss.backward()

        optimizer.step()

        if epoch % 20 == 0:
            print(
                f"[{train_seconds:2d}s] "
                f"Epoch {epoch:3d} "
                f"Loss = {loss.item():.6f}"
            )

    return model

def predict_residual(
    model,
    X,
):

    X = torch.FloatTensor(X)

    model.eval()

    with torch.no_grad():

        prediction = model(X)

    return prediction.numpy().flatten()


class OnlineResidualLearner:

    def __init__(
            self,
            lr=MLP_LEARNING_RATE,
            buffer_size=MLP_BUFFER_SIZE,
            batch_size=MLP_BATCH_SIZE,
            update_every=MLP_UPDATE_EVERY,
            start_time=MLP_START_TIME,      # seconds
            dt=0.01,
        ):

        self.start_step = int(start_time / dt)
        self.start_time = start_time
        self.model = ResidualMLP()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        self.criterion = nn.MSELoss()

        self.buffer = deque(maxlen=buffer_size)

        self.batch_size = batch_size
        self.update_every = update_every

        self.step = 0

    def predict(self, x, v, u):
        """
        Predict residual for the current state.
        """

        self.model.eval()

        inp = torch.FloatTensor([[x, v, u]])

        with torch.no_grad():
            pred = self.model(inp)

        return pred.item()

    def update(self, x, v, u, residual, t):
        """
        Online learning.

        Every timestep:
            - predict residual
            - store sample

        Every update_every timesteps:
            - sample random minibatch
            - perform one gradient update
        """


        # -----------------------------------
        # Prediction (always)
        # -----------------------------------

        self.model.eval()

        inp = torch.FloatTensor([[x, v, u]])

        with torch.no_grad():
            prediction = self.model(inp)

        residual_hat = prediction.item()

        # -----------------------------------
        # Store experience
        # -----------------------------------

        self.buffer.append(
            (
                x,
                v,
                u,
                residual,
            )
        )

        self.step += 1

        if self.step < self.start_time:
            return 0.0, None

        # -----------------------------------
        # Wait until enough data exists
        # -----------------------------------

        if len(self.buffer) < self.batch_size:
            return residual_hat, None

        # -----------------------------------
        # Only train every N timesteps
        # -----------------------------------

        if self.step % self.update_every != 0:
            return residual_hat, None

        # -----------------------------------
        # Sample random minibatch
        # -----------------------------------

        batch = random.sample(
            self.buffer,
            self.batch_size,
        )

        states = torch.FloatTensor(
            [
                [x, v, u]
                for x, v, u, _ in batch
            ]
        )

        targets = torch.FloatTensor(
            [
                [r]
                for _, _, _, r in batch
            ]
        )

        # -----------------------------------
        # Gradient update
        # -----------------------------------

        self.model.train()

        self.optimizer.zero_grad()

        predictions = self.model(states)

        loss = self.criterion(
            predictions,
            targets,
        )

        loss.backward()

        self.optimizer.step()

        return residual_hat, loss.item()
