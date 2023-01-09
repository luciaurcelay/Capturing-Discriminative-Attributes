import xgboost as xgb


class XGBClassifier:
    def __init__(self, eta, n_estimators=100, max_depth=6, colsamp_bytree=1) -> None:
        self.classifier = xgb.XGBClassifier(
            verbosity=0,
            eta=eta,
            n_estimators=n_estimators,
            max_depth=max_depth,
            colsample_bytree=colsamp_bytree,
        )
        self.params = self.create_params_dict(eta, n_estimators, max_depth, colsamp_bytree)

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

    def create_params_dict(self, eta, n_estimators, max_depth, colsamp_bytree):
        param_names = ["eta", "n_estimators", "max_depth", "colsamp_bytree"]
        params = [eta, n_estimators, max_depth, colsamp_bytree]
        return dict(zip(param_names, params))
