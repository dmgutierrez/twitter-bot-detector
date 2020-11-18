import os
import tensorflow as tf
from helper.settings import logger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import InputLayer, Dense, Activation
from tensorflow.keras.metrics import (Precision, Recall, TrueNegatives,
                                      TruePositives, FalseNegatives,
                                      FalsePositives)


class ModelConnector:
    def __init__(self):
        self.custom_metrics: dict = self.get_custom_metrics()

    @staticmethod
    def get_custom_metrics():
        custom_metrics: dict = {"precision": Precision(),
                                "recall": Recall(),
                                "true_positives": TruePositives(),
                                "true_negatives": TrueNegatives(),
                                "false_negatives": FalseNegatives(),
                                "false_positives": FalsePositives()}
        return custom_metrics

    @staticmethod
    def load_trained_model(model_directory: str, model_name: str) -> Model:
        model: Model = object.__new__(Model)
        try:
            model_path: str = os.path.join(model_directory, model_name)
            custom_objects: dict = ModelConnector.get_custom_metrics()

            # Load model
            model: Model = tf.keras.models.load_model(model_path, custom_objects, compile=False)
        except Exception as e:
            logger.error(e)
        return model

    @staticmethod
    def get_model_dimensions(model: Model, embedding_layer_name: str = "embedding") -> tuple:
        dims: tuple = (-1, -1, -1)
        try:
            # Get layers
            input_layer: InputLayer = model.get_layer(index=0)
            embedding_layer: Dense = model.get_layer(name=embedding_layer_name)
            output_layer: Activation = model.get_layer(index=-1)

            # Get dims
            dims: tuple = (input_layer.output_shape,
                           embedding_layer.output_shape,
                           output_layer.output_shape)
        except Exception as e:
            logger.error(e)
        return dims