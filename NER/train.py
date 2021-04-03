import pandas as pd
import numpy as np
from tqdm import tqdm
from torchsummary import summary
import torch
import torch.nn as nn
from BiLTSM_model import BiLSTM_Model
from utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"
import time
print(device)

sentences = pd.read_csv("Data/tokenized_sentences.csv", header=None)
labels = pd.read_csv("Data/tokenized_labels.csv", header=None)
tr_dl, val_dl, vocab_size, num_classes = get_data(sentences, labels)
model = BiLSTM_Model(vocab_size, 100, num_classes).to(device)
print("Model Loaded")


epochs = 15
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss, accu = [], []

time.sleep(1)
for epoch in range(epochs):
    print("Epoch: ", str(epoch))
    epoch_loss, epoch_accu = [], []
    time.sleep(1)
    for ix, batch in enumerate(tqdm(iter(tr_dl))):
        sentences, labels = batch
        batch_loss = train_batch(sentences, labels, model, opt, loss_fn)
        epoch_loss.append(batch_loss)

    epoch_loss = np.array(epoch_loss).mean()

    time.sleep(1)
    for ix, batch in enumerate(iter(val_dl)):
        sentence, labels = batch
        is_correct = accuracy(sentence, labels, model)
        epoch_accu.extend(is_correct)

    epoch_accu = np.mean(epoch_accu)

    print("Epoch Loss: " + str(epoch_loss))
    print("Epoch Accuracy: " + str(epoch_accu * 100)[:4] + "%\n")
    loss.append(epoch_loss)
    accu.append(epoch_accu)

torch.save(model.to("cpu").state_dict(), "Saved_Model/my_model.pth")
print("Model Saved")
