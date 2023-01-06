import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input


class MLP:
    def __init__(self, input_dim) -> None:
        self.input_dim = input_dim
        self.output_dim = 1
        self.loss = keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer = "adam"
        self.metrics = ["accuracy", "Precision", "Recall"]
        self.model = self.build_model()
        self.epochs = 50

    def build_model(self):
        # Architecture from
        model = keras.Sequential(
            [
                Input(shape=(self.input_dim,)),
                layers.Dense(12, activation="relu", name="layer1"),
                layers.Dropout(0.3),
                layers.Dense(12, activation="relu", name="layer2"),
                layers.Dense(100, activation="relu", name="layer3"),
                layers.Dense(200, activation="relu", name="layer4"),
                layers.Dense(self.output_dim, name="layer5", activation="relu"),
            ]
        )
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def print_summary(self):
        x = tf.ones((1, self.input_dim))
        self.model(x)
        print(self.model.summary())

    def train_classifier(self, X_train, y_train, X_val):
        self.model.fit(X_train, y_train, epochs=self.epochs)
        # Make predictions
        predictions = self.model.predict(X_val)
        predictions[predictions <= 0.5] = 0
        predictions[predictions > 0.5] = 1

        return predictions


# Layer type Output Shape Param #
# Dense1 (None, 12) 132
# Dropout1 (None, 12) 0
# Dense2 (None, 12) 156
# Dense3 (None, 100) 1300
# Dense4 (None, 200) 20200
# Dense5 (None, 1) 201
