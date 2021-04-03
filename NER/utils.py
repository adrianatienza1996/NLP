import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"


class GMBDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __getitem__(self, ix):
        x, y = self.sentences[ix], self.labels[ix]
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        return x, y

    def __len__(self): return len(self.sentences)


def get_data(sentences, labels, batch_size=64, test_ratio=0.2):
    sentences = np.array(sentences)
    labels = np.array(labels)
    idx = np.random.uniform(0, 1, sentences.shape[0]) > test_ratio
    train_sentences = sentences[idx]
    train_labels = labels[idx]
    test_sentences = sentences[~idx]
    test_labels = labels[~idx]

    train = GMBDataset(train_sentences, train_labels)
    trn_dl = DataLoader(train, batch_size=batch_size, shuffle=True)

    val = GMBDataset(test_sentences, test_labels)
    val_dl = DataLoader(val, batch_size=len(val), shuffle=False)
    return trn_dl, val_dl


def train_batch(sentences, labels, model, opt, loss_fn):
    model.train()
    prediction = model(sentences)
    loss = loss_fn(prediction, labels)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


@torch.no_grad()
def accuracy(sentences, labels, model):
    model.eval()
    prediction = model(sentences)
    _, argmaxes = prediction.max(-1)
    is_correct = prediction == labels
    return is_correct.cpu().numpy().tolist()