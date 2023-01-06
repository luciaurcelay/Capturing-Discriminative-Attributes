import xgboost as xgb
import pandas as pd


class XGBClassifier:
    def __init__(self) -> None:
        self.classifier = xgb.XGBClassifier()

    def train_classifier(self, X_train, y_train, X_val) -> None:
        X_train = self.model_specific_preprocessing(X_train)
        # X_val = X_val.drop(columns=["index"])
        # Fit
        self.classifier.fit(X_train, y_train)
        # Make predictions
        predictions = self.classifier.predict(X_val)

        return predictions

    def model_specific_preprocessing(self, data):
        data = data.astype(float)
        return data
