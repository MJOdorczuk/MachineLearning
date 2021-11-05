from __future__ import annotations

import torch
from sklearn.metrics import r2_score


def make_and_train_torch_nn(X, z, lr=0.1, epochs=20):
    """Make a simple pytorch network to compare with."""

    X = torch.as_tensor(X, dtype=torch.float)
    z = torch.as_tensor(z, dtype=torch.float)

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Sigmoid(),
        torch.nn.Linear(4, 8),
        torch.nn.Sigmoid(),
        torch.nn.Linear(8, 16),
        torch.nn.Sigmoid(),
        torch.nn.Linear(16, 1),
    )

    cost_func = torch.nn.MSELoss()

    all_loss = []
    all_r2 = []

    for e in range(epochs):
        print(f"Epoch {e}")
        pred = model(X)
        loss = cost_func(pred, z)
        all_loss.append(loss.detach().numpy().copy())
        all_r2.append(
            r2_score(z.detach().numpy(), pred.detach().numpy())
        )
        print(f"Loss {loss:.2f} R2 {all_r2[-1]:.2f}")
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad

    return model, all_loss, all_r2
