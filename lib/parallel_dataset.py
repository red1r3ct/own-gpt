import numpy as np
from torch.utils.data import DataLoader


class ParallelDataset:
    def __init__(self, path, context_window):
        all_data = np.load(path, allow_pickle=True)
        self.data = all_data[split].astype("int64")
        self.len = len(self.data)
        self.context_window = context_window

    def __getitem__(self, key: int):
        i = np.random.RandomState().randint(self.len - self.context_window - 1)
        x_doc_emb = self.data[key][i : i + self.context_window]
        y_doc_emb = self.data[key][i + 1 : i + self.context_window + 1]
        return x_doc_emb, y_doc_emb

    def __len__(self):
        return self.len


def make_dataloader(
    path: str,
    block_size: int,
    batch_size: int,
    num_workers=4,
) -> DataLoader:
    return (
        DataLoader(
            ParallelDataset(path, block_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
    )
