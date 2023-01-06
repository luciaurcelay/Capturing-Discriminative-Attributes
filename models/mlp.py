import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MLP:
    def __init__(self, input_dim) -> None:
        self.input_dim = input_dim
        self.output_dim = 1
        self.model = self.build_model()

    def build_model(self):
        # Architecture from
        model = keras.Sequential(
            [
                layers.Dense(12, activation="relu", name="layer1"),
                layers.Dropout(0.3),
                layers.Dense(12, activation="relu", name="layer2"),
                layers.Dense(100, activation="relu", name="layer3"),
                layers.Dense(200, activation="relu", name="layer4"),
                layers.Dense(self.output_dim, name="layer5"),
            ]
        )
        return model

    def print_summary(self):
        x = tf.ones((1, self.input_dim))
        self.model(x)
        print(self.model.summary())


# Layer type Output Shape Param #
# Dense1 (None, 12) 132
# Dropout1 (None, 12) 0
# Dense2 (None, 12) 156
# Dense3 (None, 100) 1300
# Dense4 (None, 200) 20200
# Dense5 (None, 1) 201
