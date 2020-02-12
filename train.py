"""
Script to train model.
"""
import logging
import os
import pickle

from bedrock_client.bedrock.api import BedrockApi
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession

from utils.constants import FEATURE_COLS, TARGET_COL
from utils.preprocess import generate_features

LR = float(os.getenv("LR"))
NUM_LEAVES = int(os.getenv("NUM_LEAVES"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS"))
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME")


def compute_log_metrics(gbm, x_val, y_val):
    """Compute and log metrics."""
    print("\tEvaluating using validation data")
    y_prob = gbm.predict_proba(x_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)
    print("Accuracy = {:.6f}".format(acc))
    print("Precision = {:.6f}".format(precision))
    print("Recall = {:.6f}".format(recall))
    print("F1 score = {:.6f}".format(f1_score))
    print("AUC = {:.6f}".format(auc))
    print("Average precision = {:.6f}".format(avg_prc))

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy 2", acc)
    bedrock.log_metric("Precision 2", precision)
    bedrock.log_metric("Recall 2", recall)
    bedrock.log_metric("F1 score 2", f1_score)
    bedrock.log_metric("AUC 2", auc)
    bedrock.log_metric("Avg precision 2", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(),
                           y_prob.flatten().tolist())


def main():
    """Train pipeline"""
    print("\tGenerating features")
    with SparkSession.builder.appName("FeatureGeneration").getOrCreate() as spark:
        spark.sparkContext.setLogLevel("FATAL")
        model_data = generate_features(spark).toPandas()

    print("\tSplitting train and validation data")
    x_train, x_val, y_train, y_val = train_test_split(
        model_data[FEATURE_COLS],
        model_data[TARGET_COL],
        test_size=0.2,
    )

    print("\tTrain model")
    gbm = lgb.LGBMClassifier(
        num_leaves=NUM_LEAVES,
        learning_rate=LR,
        n_estimators=N_ESTIMATORS,
    )
    gbm.fit(x_train, y_train)
    compute_log_metrics(gbm, x_val, y_val)

    print("\tSaving model")
    with open("/artefact/" + OUTPUT_MODEL_NAME, "wb") as model_file:
        pickle.dump(gbm, model_file)


if __name__ == "__main__":
    main()
