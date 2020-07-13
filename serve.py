"""
Script for serving.
"""
import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from utils.constants import AREA_CODES, STATES, SUBSCRIBER_FEATURES


def pre_process(http_body):
    """Predict churn probability given subscriber_features.

    Args:
        subscriber_features (dict)
        model

    Returns:
        churn_prob (float): churn probability
    """
    subscriber_features = json.loads(http_body)
    row_feats = list()
    for col in SUBSCRIBER_FEATURES:
        row_feats.append(subscriber_features[col])

    for area_code in AREA_CODES:
        if subscriber_features["Area_Code"] == area_code:
            row_feats.append(1)
        else:
            row_feats.append(0)

    for state in STATES:
        if subscriber_features["State"] == state:
            row_feats.append(1)
        else:
            row_feats.append(0)

    return np.array(row_feats).reshape(1, -1)


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load("/artefact/torch_model.pkl").to(self.device)
        self.model.eval()

    def predict(self, features):
        return F.sigmoid(self.model(torch.tensor(np.array(features)).float().to(self.device))).cpu().detach().item()
