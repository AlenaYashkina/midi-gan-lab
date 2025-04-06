from torch import nn


class InstrumentClassifier(nn.Module):
    def __init__(self, num_instruments: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None
        self.num_instruments = num_instruments

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], self.num_instruments).to(x.device)
        return self.fc(x)
