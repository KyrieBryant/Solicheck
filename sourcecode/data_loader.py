from torch.utils import data
import torch

class SoliCheckDataset(data.Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]

        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class LongTextDataset(data.Dataset):
    def __init__(self, tokenized_texts, labels):
        self.tokenized_texts = tokenized_texts
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long) # 返
 