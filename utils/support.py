import torch
import numpy as np

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

# Training loop
early_stopping = EarlyStopping(patience=5, delta=1e-4)


