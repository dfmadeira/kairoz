import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)

        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

        self.act = nn.SiLU()  # smooth activation

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.ln2(x)

        return self.act(x + residual)


class ThetaEstimator(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()

        # input projection (feature lifting)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )

        # deep feature extractor
        self.blocks = nn.Sequential(
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )

        # bottleneck / regularization
        self.middle = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )

        # output head (θ estimation)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),  # θ1, θ2, θ3
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x


def train_model():

    X, Y = generate_dataset()

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    model = ThetaEstimator(X.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        pred = model(X)
        loss = loss_fn(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch {epoch} loss {loss.item():.6f}")

    torch.save(model.state_dict(), "theta_model.pt")

    return model


def load_model(input_dim):
    model = ThetaEstimator(input_dim)
    model.load_state_dict(torch.load("theta_model.pt"))
    model.eval()
    return model
