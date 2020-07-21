"""
Script for serving.
"""
import json
import pickle

import numpy as np
from transformers import pipeline


def pre_process(http_body):
    return json.loads(http_body)["query"]


class Model:
    def __init__(self):
        self.model = pipeline('sentiment-analysis')

    def predict(self, features):
        return self.model(features)[0]
