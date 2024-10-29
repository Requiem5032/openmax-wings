import torch

from sklearn.metrics import accuracy_score, f1_score


def accuracy(preds, target):
    return torch.tensor(accuracy_score(target.cpu().numpy(), preds.cpu().numpy().argmax(1)))


def macro_f1(preds, target):
    return torch.tensor(f1_score(target.cpu().numpy(), preds.cpu().argmax(1), average='macro'))
