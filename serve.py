"""
Script for serving.
"""
import json
import pickle

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models


def pre_process(http_body):
    """Predict churn probability given subscriber_features.

    Args:
        subscriber_features (dict)
        model

    Returns:
        churn_prob (float): churn probability
    """
    features = np.array(json.loads(http_body)["image"]) / 255
    features_t = torch.tensor(features)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return normalize(features_t).unsqueeze(dim=0)


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, features):
        return torch.argmax(self.model(features).squeeze()).cpu().detach().item()
