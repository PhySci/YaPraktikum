import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import logging
import pandas as pd
from tqdm import tqdm

_logger = logging.getLogger(__name__)


class TSDataset(Dataset):

    def __init__(self, pth, window_size=128):
        self._df = pd.read_csv(pth)
        self._window = window_size

    def __len__(self):
        return self._df.shape[0] - self._window - 1

    def __getitem__(self, item):
        x = self._df.iloc[item:item+self._window]["power"]
        y = self._df.iloc[item+self._window]["power"]
        return torch.FloatTensor(x.values), torch.FloatTensor([y])


class LSTM(nn.Module):

    def __init__(self, n_cells=64):
        super(LSTM, self).__init__()
        self._lstm = nn.LSTM(input_size=128, hidden_size=n_cells, num_layers=1, batch_first=True)
        self._fc = nn.Linear(64, 1)

    def forward(self, x):
        y, (h, c) = self._lstm(x)
        return self._fc(y)


def main():

    ds = TSDataset(pth="./data/household_power_consumption_v2.txt")
    dl = DataLoader(ds, batch_size=64)

    model = LSTM()
    opt = Adam(model.parameters(), lr=1e-3)

    lf = nn.MSELoss()

    # epoch
    model.train()

    # add several epochs

    for x, y in tqdm(dl):
        opt.zero_grad()
        y_pr = model(x)
        loss = lf(y, y_pr)
        loss.backward()
        opt.step()
        print(loss.item())

    # add validation
    # add logging
    # add save model


def test_dataset():
    d = TSDataset(pth="./data/household_power_consumption_v2.txt")

    for x, y in d:
        print(x, y)

if __name__ == "__main__":
    main()
    #test_dataset()