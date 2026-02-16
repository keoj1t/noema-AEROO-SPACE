import torch
import torch.nn as nn
import numpy as np

FEATURES = 3

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, FEATURES)
        )

    def forward(self, x):
        return self.net(x)

def generate_normal_data(n=2000):
    data = []
    for _ in range(n):
        data.append([
            35 + np.random.uniform(-0.3, 0.3),
            28 + np.random.uniform(-0.1, 0.1),
            max(0.0, np.random.uniform(-0.05, 0.05))
        ])
    return torch.tensor(data, dtype=torch.float32)

def main():
    model = AutoEncoder()
    X = generate_normal_data()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(25):
        opt.zero_grad()
        loss = loss_fn(model(X), X)
        loss.backward()
        opt.step()
        print(f"Epoch {epoch} | loss {loss.item():.6f}")

    model.eval()
    dummy = torch.randn(1, FEATURES)

    torch.onnx.export(
        model,
        dummy,
        "ae.onnx",
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17
    )

    print("✅ ae.onnx exported")

if __name__ == "__main__":
    main()
