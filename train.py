"""
Script to train model.
"""
import logging
import os
import pickle

import numpy as np
import pandas as pd
from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import torch.nn.functional as F

from utils.constants import FEATURE_COLS, TARGET_COL

FEATURES_DATA = os.path.join(os.getenv("TEMP_DATA_BUCKET"),
                             os.getenv("FEATURES_DATA"))
LR = float(os.getenv("LR"))
NUM_LEAVES = int(os.getenv("NUM_LEAVES"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS"))
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME")


class TorchModel(nn.Module):
    def __init__(self, input_dim):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_log_metrics(clf, x_val, y_val, device):
    """Compute and log metrics."""
    print("\tEvaluating using validation data")
    y_prob = F.sigmoid(clf(torch.tensor(np.array(x_val)).float().to(device))).cpu().detach().numpy()
    y_pred = (y_prob > 0.5).astype(int)

    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    roc_auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)

    print(f"Accuracy          = {acc:.6f}")
    print(f"Precision         = {precision:.6f}")
    print(f"Recall            = {recall:.6f}")
    print(f"F1 score          = {f1_score:.6f}")
    print(f"ROC AUC           = {roc_auc:.6f}")
    print(f"Average precision = {avg_prc:.6f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1_score)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(),
                           y_prob.flatten().tolist())


def main():
    """Train pipeline"""
    model_data = pd.read_csv(FEATURES_DATA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\tSplitting train and validation data")
    x_train, x_val, y_train, y_val = train_test_split(
        model_data[FEATURE_COLS],
        model_data[TARGET_COL],
        test_size=0.2,
    )

    print("\tTrain model")
    clf = TorchModel(
        input_dim=x_train.shape[1]
    ).to(device)
    optimizer = optim.Adam(clf.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    batchsize = 16
    for t in range(len(x_train) // batchsize):
        optimizer.zero_grad()
        batch_x = x_train[t * batchsize:(t + 1) * batchsize]
        batch_y = y_train[t * batchsize:(t + 1) * batchsize]
        logits = clf(torch.tensor(np.array(batch_x)).float().to(device))
        loss = loss_fn(logits, torch.tensor(np.array(batch_y)).float().to(device).view(-1, 1))
        loss.backward()
        optimizer.step()
    compute_log_metrics(clf, x_val, y_val, device)

    print("\tComputing metrics")
    selected = np.random.choice(model_data.shape[0], size=1000, replace=False)
    features = model_data[FEATURE_COLS].iloc[selected]
    inference = F.sigmoid(clf(torch.tensor(np.array(features)).float().to(device))).cpu().detach().numpy()

    ModelMonitoringService.export_text(
        features=features.iteritems(),
        inference=inference.tolist(),
    )

    print("\tSaving model")
    torch.save(clf.state_dict(), "/artefact/" + OUTPUT_MODEL_NAME)


if __name__ == "__main__":
    main()
