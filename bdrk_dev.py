import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, List, MutableMapping, Optional
from uuid import UUID, uuid4

import requests
from google.cloud import bigquery
from prometheus_client import Histogram


@dataclass(frozen=True)
class Prediction:
    entity_id: UUID
    features: List[float]
    requestBody: str
    output: float
    server_id: str
    created_at: datetime = datetime.now(tz=timezone.utc)


class PredictionStore:
    def __init__(self):
        # TODO: replace in memory store with BigQuery
        self._store: MutableMapping[UUID, Prediction] = {}
        # TODO: Support AWS native storage
        self._client = bigquery.Client()
        self._table = self._client.get_table("span-staging.expt_prediction_store.prediction_v1")
        # Uses context var to handle context between multiple web handlers
        self._scope = ContextVar("scope")
        # TODO: fetch tracked features from run metrics
        self._tracked = {"very_important_feature_0": [5.0, 10.0, 15.0]}
        # Load feature vector bins
        self._histogram = {
            name: Histogram(
                name=f"model_feature_{name}_value",
                documentation=f"Serving time values for model feature: {name}",
                # Only need to label by server / endpoint id when using pushgateway
                # labelnames=("server_id"),
                buckets=tuple(bins),
            )
            for name, bins in self._tracked.items()
        }

    def save(self, prediction: Prediction):
        """
        Stores the prediction asynchronously to BigQuery.

        :param prediction: The completed prediction
        :type prediction: Prediction
        """
        data = asdict(prediction)
        data["entity_id"] = str(prediction.entity_id)
        # TODO: Supports bytes type which is not json serializable
        errors = self._client.insert_rows(self._table, [data])
        if errors:
            print(f"Error adding row: {errors}")
        else:
            print(f"New row added: {data}")

        # Export feature value for scraping
        for name, _ in self._tracked.items():
            index = int(name.split("_")[-1])
            value = prediction.features[index]
            self._histogram[name].observe(value)


    def load(self, key: UUID) -> Prediction:
        """
        Loads a prediction by its primary key.

        :param key: The primary key of the prediction.
        :type key: UUID
        :return: The past prediction.
        :rtype: Prediction
        """
        return self._store[key]

    def log(self, **kwargs):
        """
        Logs partial attributes to the currently active prediction.

        :raises RuntimeError: When no active scope is available.
        """
        active = self._scope.get()
        active.update(**kwargs)

    @contextmanager
    def activate(self) -> UUID:
        """
        Activates a prediction scope.

        :raises RuntimeError: When an active scope already exists.
        :yield: The prediction ID of the active scope
        :rtype: UUID
        """
        key = uuid4()
        token = self._scope.set({
            "server_id": os.environ["SERVER_ID"],
            "entity_id": key
        })

        try:
            yield key
        finally:
            active = self._scope.get()
            self._scope.reset(token)
            self.save(Prediction(**active))


store = PredictionStore()


def track(func: Callable) -> Callable:
    """
    Middleware for automatically tracking predict function output.

    :param func: The predict function to decorate
    :type func: Callable
    :return: The decorated function
    :rtype: Callable
    """

    def wrapper(*args, **kwargs):
        with store.activate() as key:
            resp = func(*args, **kwargs)
            # resp.header["X-Prediction-ID"] = key
            store.log(output=resp)
            return resp

    return wrapper


def log_feature_histogram(index: int, bins: List[float], name: Optional[str] = None):
    """
    Logs the histogram bins for a feature at the specified index of the feature vector. If no name
    is specified, a default value of feature_{index} will be used.

    :param index: Index of the logged feature in the feature vector
    :type index: int
    :param bins: The upperbound of each bin in ascending order
    :type bins: List[float]
    :param name: An optional feature name for visualization
    :type name: Optional[str]
    """
    if len(bins) > 10:
        raise RuntimeError("Bin size cannot be more than 10")

    # Calls run metric api internally for now
    domain = os.environ.get("BEDROCK_API_DOMAIN", "https://api.bdrk.ai")
    headers = {
        "X-Bedrock-Api-Token": os.environ["BEDROCK_API_TOKEN"]
    }
    data = {
        "key": f"{name or 'feature'}_{index}",
        "value": ",".join(bins)
    }
    with requests.post(f"{domain}/internal/run/metrics", data=data, headers=headers) as resp:
        print(resp)