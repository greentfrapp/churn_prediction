"""
Script for serving.
"""
import json
import pickle

import numpy as np
import tensorflow as tf

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
        self.model = tf.keras.models.load_model('/artefact/my_model')

    def predict(self, features):
        return self.model(np.array(features)).numpy()
